// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package service_test

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/dsx-ai-factory/nvcf/src/compute-plane-services/request-trace-uploader/backend"
	"github.com/dsx-ai-factory/nvcf/src/compute-plane-services/request-trace-uploader/backend/backendtest"
	"github.com/dsx-ai-factory/nvcf/src/compute-plane-services/request-trace-uploader/config"
	"github.com/dsx-ai-factory/nvcf/src/compute-plane-services/request-trace-uploader/format"
	"github.com/dsx-ai-factory/nvcf/src/compute-plane-services/request-trace-uploader/internal/journalfs"
	"github.com/dsx-ai-factory/nvcf/src/compute-plane-services/request-trace-uploader/journal"
	"github.com/dsx-ai-factory/nvcf/src/compute-plane-services/request-trace-uploader/record"
	"github.com/dsx-ai-factory/nvcf/src/compute-plane-services/request-trace-uploader/service"
	"github.com/dsx-ai-factory/nvcf/src/compute-plane-services/request-trace-uploader/telemetry"
	"github.com/dsx-ai-factory/nvcf/src/compute-plane-services/request-trace-uploader/transform"
	"github.com/dsx-ai-factory/nvcf/src/compute-plane-services/request-trace-uploader/uploadertest"
)

// hookedFormat is a backend that writes gzip JSON lines through a writer
// that calls onWrite before each record, so a test can hold a prepare in the
// middle of a segment.
type hookedFormat struct {
	*backendtest.Fake
	onWrite func(target format.Target, r transform.Record) error
}

func (h hookedFormat) Format() backend.Format { return backend.FormatGzipJSONL }

func (h hookedFormat) NewWriter(target format.Target) (format.Writer, error) {
	w, err := format.GzipJSONL{}.NewWriter(target)
	if err != nil {
		return nil, err
	}
	return hookedWriter{Writer: w, target: target, onWrite: h.onWrite}, nil
}

type hookedWriter struct {
	format.Writer
	target  format.Target
	onWrite func(format.Target, transform.Record) error
}

func (w hookedWriter) Write(r transform.Record) error {
	if err := w.onWrite(w.target, r); err != nil {
		return err
	}
	return w.Writer.Write(r)
}

func workersEnv(n string) map[string]string {
	return map[string]string{config.EnvPrepareWorkers: n}
}

// The first record of each segment waits until the other segment's first
// record arrives: only two prepares running at once get past it.
func TestTwoWorkersPrepareTwoSegmentsAtOnce(t *testing.T) {
	spool, closed := spoolWith(t,
		[]string{uploadertest.RequestEnd("a", 1)},
		[]string{uploadertest.RequestEnd("b", 2)},
	)
	var arrived sync.WaitGroup
	arrived.Add(2)
	both := make(chan struct{})
	go func() { arrived.Wait(); close(both) }()
	var once sync.Map
	client := hookedFormat{Fake: batchingFake(), onWrite: func(target format.Target, _ transform.Record) error {
		if _, seen := once.LoadOrStore(target.Segment.Index, true); seen {
			return nil
		}
		arrived.Done()
		select {
		case <-both:
			return nil
		case <-time.After(5 * time.Second):
			return errors.New("the other segment was not prepared at the same time")
		}
	}}
	uploader := newUploader(t, spool, client, service.Options{}, workersEnv("2"))
	if err := uploader.Pass(context.Background()); err != nil {
		t.Fatalf("Pass() error = %v", err)
	}
	if err := uploader.Close(); err != nil {
		t.Fatal(err)
	}
	for _, path := range closed {
		if !gone(path) {
			t.Errorf("%s was not exported and cleaned up", filepath.Base(path))
		}
	}
}

// preparedOrder lists segment indexes by when the journal recorded them
// prepared.
func preparedOrder(t *testing.T, spool *uploadertest.Spool) []journal.Segment {
	t.Helper()
	store, err := journalfs.Open(context.Background(), journalfs.Config{
		SpoolDir: spool.Dir,
		StateDir: filepath.Join(spool.Dir, "request-trace-uploader-state"),
		Holder:   "inspector",
		LockWait: time.Second,
	}, journalfs.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	return store.Segments()
}

// The first segment is the slowest to prepare, yet the journal records the
// segments prepared in their order, as one worker would.
func TestWorkersRecordPreparedSegmentsInSegmentOrder(t *testing.T) {
	var segments [][]string
	for i := range 6 {
		segments = append(segments, []string{uploadertest.RequestEnd(strings.Repeat("r", i+1), uint64(i))})
	}
	spool, _ := spoolWith(t, segments...)
	client := hookedFormat{Fake: batchingFake(), onWrite: func(target format.Target, _ transform.Record) error {
		if target.Segment.Index == 0 {
			time.Sleep(300 * time.Millisecond)
		}
		return nil
	}}
	client.Script(backendtest.SubmitStep{Outcome: backendtest.Refuse, Err: errors.New("hold everything prepared")})
	uploader := newUploader(t, spool, client, service.Options{}, workersEnv("4"))
	_ = uploader.Pass(context.Background())
	if err := uploader.Close(); err != nil {
		t.Fatal(err)
	}
	var last time.Time
	for _, seg := range preparedOrder(t, spool) {
		if seg.Index >= 6 {
			continue
		}
		if seg.PreparedAt.IsZero() {
			t.Fatalf("segment %d was not prepared (state %s)", seg.Index, seg.State)
		}
		if seg.PreparedAt.Before(last) {
			t.Fatalf("segment %d was recorded prepared before an earlier segment", seg.Index)
		}
		last = seg.PreparedAt
	}
}

// A line past the memory step held by one prepare leaves no room in a small
// budget for another such line, so the second prepare waits, and the wait
// is counted. The first prepare's long record waits in the writer until the
// wait is counted, so the two lines are held at once.
func TestAPrepareWaitingForLineMemoryIsCounted(t *testing.T) {
	long := func(id string) string {
		return `{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":1,"request":{"request_id":"` +
			id + strings.Repeat("z", record.DefaultMemoryStep+(1<<20)) + `"}}`
	}
	spool, closed := spoolWith(t, []string{long("a")}, []string{long("b")})
	metrics, err := telemetry.NewMemory("fake")
	if err != nil {
		t.Fatal(err)
	}
	var first sync.Once
	client := hookedFormat{Fake: batchingFake(), onWrite: func(_ format.Target, r transform.Record) error {
		var err error
		first.Do(func() {
			deadline := time.Now().Add(10 * time.Second)
			for {
				if waits, _ := metrics.Value(telemetry.PrepareMemoryWaits, nil); waits >= 1 {
					return
				}
				if time.Now().After(deadline) {
					err = errors.New("the other prepare never waited for line memory")
					return
				}
				time.Sleep(5 * time.Millisecond)
			}
		})
		return err
	}}
	env := workersEnv("2")
	env[config.EnvPrepareMemoryBytes] = "67108864"
	uploader := newUploader(t, spool, client, service.Options{Metrics: metrics}, env)
	if err := uploader.Pass(context.Background()); err != nil {
		t.Fatalf("Pass() error = %v", err)
	}
	if err := uploader.Close(); err != nil {
		t.Fatal(err)
	}
	if waits, _ := metrics.Value(telemetry.PrepareMemoryWaits, nil); waits != 1 {
		t.Fatalf("prepare_memory_waits = %v, want one wait", waits)
	}
	for _, path := range closed {
		if !gone(path) {
			t.Errorf("%s was not exported after the wait", filepath.Base(path))
		}
	}
}

func TestPrepareMemoryWaitsStartsAtZero(t *testing.T) {
	spool, _ := spoolWith(t)
	metrics, err := telemetry.NewMemory("fake")
	if err != nil {
		t.Fatal(err)
	}
	newUploader(t, spool, batchingFake(), service.Options{Metrics: metrics}, nil)
	if waits, ok := metrics.Value(telemetry.PrepareMemoryWaits, nil); !ok || waits != 0 {
		t.Fatalf("prepare_memory_waits = %v, %v; want 0 before any prepare", waits, ok)
	}
}

// watchedSubmits closes first when the backend receives its first submit.
type watchedSubmits struct {
	hookedFormat
	first chan struct{}
	once  sync.Once
}

func (w *watchedSubmits) Submit(ctx context.Context, request backend.SubmitRequest) (string, error) {
	w.once.Do(func() { close(w.first) })
	return w.hookedFormat.Submit(ctx, request)
}

// Batches hold two segments. The third segment's prepare waits until the
// backend has a submit, so it finishes only if the first full batch is sent
// while intake is still preparing.
func TestAFullBatchIsSubmittedWhilePreparingGoesOn(t *testing.T) {
	var segments [][]string
	for i := range 4 {
		segments = append(segments, []string{uploadertest.RequestEnd(strings.Repeat("s", i+1), uint64(i))})
	}
	spool, closed := spoolWith(t, segments...)
	fake := backendtest.NewFake()
	fake.SetCapabilities(backend.Capabilities{
		AcceptedFormats: []backend.Format{backend.FormatGzipJSONL},
		Exports:         true,
		MaxBatchFiles:   2,
	})
	client := &watchedSubmits{first: make(chan struct{})}
	client.hookedFormat = hookedFormat{Fake: fake, onWrite: func(target format.Target, _ transform.Record) error {
		if target.Segment.Index != 2 {
			return nil
		}
		select {
		case <-client.first:
			return nil
		case <-time.After(5 * time.Second):
			return errors.New("no batch was submitted while this segment was prepared")
		}
	}}
	uploader := newUploader(t, spool, client, service.Options{}, nil)
	if err := uploader.Pass(context.Background()); err != nil {
		t.Fatalf("Pass() error = %v", err)
	}
	if err := uploader.Close(); err != nil {
		t.Fatal(err)
	}
	for _, path := range closed {
		if !gone(path) {
			t.Errorf("%s was not exported and cleaned up", filepath.Base(path))
		}
	}
	jobs := fake.Jobs()
	if len(jobs) != 2 {
		t.Fatalf("jobs = %d, want two full batches of two", len(jobs))
	}
	for _, job := range jobs {
		if len(job.Files) != 2 {
			t.Fatalf("job %s has %d files, want full batches only", job.ID, len(job.Files))
		}
	}
}

// Twenty segments and jobs of sixteen: the first job fills at the
// prepare-ahead bound. Its upload waits until the seventeenth segment is
// being prepared, so it finishes only if intake waits for the batch to be
// recorded and goes on, instead of ending the pass at the bound.
func TestABacklogKeepsPreparingPastTheBoundWhileAFullBatchUploads(t *testing.T) {
	var segments [][]string
	for i := range 20 {
		segments = append(segments, []string{uploadertest.RequestEnd(strings.Repeat("u", i+1), uint64(i))})
	}
	spool, closed := spoolWith(t, segments...)
	later := make(chan struct{})
	var once sync.Once
	fake := backendtest.NewFake()
	fake.SetCapabilities(backend.Capabilities{
		AcceptedFormats: []backend.Format{backend.FormatGzipJSONL},
		Exports:         true,
		MaxBatchFiles:   16,
	})
	client := &gatedSubmits{hookedFormat: hookedFormat{Fake: fake, onWrite: func(target format.Target, _ transform.Record) error {
		if target.Segment.Index >= 16 {
			once.Do(func() { close(later) })
		}
		return nil
	}}, gate: later}
	uploader := newUploader(t, spool, client, service.Options{}, workersEnv("2"))
	if err := uploader.Pass(context.Background()); err != nil {
		t.Fatalf("Pass() error = %v", err)
	}
	if err := uploader.Close(); err != nil {
		t.Fatal(err)
	}
	for _, path := range closed {
		if !gone(path) {
			t.Errorf("%s was not exported in the pass", filepath.Base(path))
		}
	}
}

// gatedSubmits holds its first submit until gate closes.
type gatedSubmits struct {
	hookedFormat
	gate  chan struct{}
	first sync.Once
}

func (g *gatedSubmits) Submit(ctx context.Context, request backend.SubmitRequest) (string, error) {
	var err error
	g.first.Do(func() {
		select {
		case <-g.gate:
		case <-time.After(5 * time.Second):
			err = errors.New("no segment past the prepare-ahead bound was prepared during the upload")
		}
	})
	if err != nil {
		return "", err
	}
	return g.hookedFormat.Submit(ctx, request)
}
