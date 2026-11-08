// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package service

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"time"

	"github.com/dsx-ai-factory/nvcf/src/compute-plane-services/request-trace-uploader/internal/diskguard"
)

// backlogSubmits coordinates intake with the full-batch submits that run
// beside it. Intake reports each segment it records prepared; the submitter
// runs a round per report and reports each batch it records, so intake can
// wait at the prepare-ahead bound for the next batch instead of ending the
// pass.
type backlogSubmits struct {
	// wake holds one pending request for a round. Only intake sends.
	wake chan struct{}

	mu sync.Mutex
	// changed is closed and replaced when a round starts or ends, or a batch
	// is recorded.
	changed chan struct{}
	// prepared counts segments intake recorded prepared; seen is that count
	// when the latest round started.
	prepared, seen uint64
	running        bool
}

func newBacklogSubmits() *backlogSubmits {
	return &backlogSubmits{wake: make(chan struct{}, 1), changed: make(chan struct{})}
}

func (s *backlogSubmits) notePrepared() {
	s.mu.Lock()
	s.prepared++
	s.mu.Unlock()
	s.wakeUp()
}

func (s *backlogSubmits) wakeUp() {
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

func (s *backlogSubmits) broadcastLocked() {
	close(s.changed)
	s.changed = make(chan struct{})
}

func (s *backlogSubmits) roundStarted() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.running = true
	s.seen = s.prepared
	s.broadcastLocked()
}

func (s *backlogSubmits) roundEnded() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.running = false
	s.broadcastLocked()
}

func (s *backlogSubmits) batched() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.broadcastLocked()
}

// settled reports that no round runs or is asked for and the latest round
// saw every prepared segment, so no submit will record another batch until
// intake prepares more. It also returns the channel the next change closes.
func (s *backlogSubmits) settled() (<-chan struct{}, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.changed, !s.running && s.seen >= s.prepared && len(s.wake) == 0
}

// intakeSubmitting runs intake and, beside it, submits each batch that
// fills up from what intake prepares, so a backlog is sent while it is
// prepared instead of after. A batch that is not full waits for the end of
// the pass, as before. The caller holds the intake lock; submits take the
// submit lock, so they never run beside another submit.
func (u *Uploader) intakeSubmitting(ctx context.Context) error {
	s := newBacklogSubmits()
	u.backlog = s
	submitted := make(chan error, 1)
	go func() {
		var errs []error
		for range s.wake {
			s.roundStarted()
			errs = append(errs, u.locked(&u.submitMu, func() error { return u.submitFull(ctx) }))
			s.roundEnded()
		}
		submitted <- errors.Join(errs...)
	}()
	err := u.intake(ctx, false)
	close(s.wake)
	err = errors.Join(err, <-submitted)
	u.backlog = nil
	return err
}

// roomToPrepare reports whether one more segment of size bytes may be
// prepared. At the prepare-ahead bound, while full batches are submitted
// beside intake, it waits for the next batch to be recorded instead of
// ending the pass. It gives up when the submitter has seen every prepared
// segment and recorded nothing, and when the uploader drains. It logs a
// pause.
func (u *Uploader) roomToPrepare(ctx context.Context, size int64, pool *preparePool) (bool, error) {
	for {
		decision, err := u.allowPrepare(size)
		if err != nil || decision.Allowed {
			return decision.Allowed, err
		}
		waitable := decision.Reason == diskguard.PausePrepareAhead && u.backlog != nil && !u.draining.Load()
		var changed <-chan struct{}
		if waitable {
			// Finished prepares count, and can fill the next batch.
			pool.recordReady()
			var settled bool
			changed, settled = u.backlog.settled()
			waitable = !settled
		}
		if !waitable {
			slog.Warn("preparing paused to protect the spool; prepared work is still sent", "reason", decision.Reason)
			return false, nil
		}
		u.backlog.wakeUp()
		select {
		case <-changed:
		case <-ctx.Done():
			return false, ctx.Err()
		case <-time.After(time.Second):
			// Look at draining again.
		}
	}
}
