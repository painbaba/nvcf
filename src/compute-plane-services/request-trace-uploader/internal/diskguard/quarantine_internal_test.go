// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package diskguard

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
	"time"
)

func TestAdmitLinksNewDirectoriesBeforeMovingFiles(t *testing.T) {
	spool := t.TempDir()
	dir := filepath.Join(spool, ".rtu-quarantine")
	source := filepath.Join(spool, "request-trace.000003.jsonl.gz")
	if err := os.WriteFile(source, []byte("segment"), 0o640); err != nil {
		t.Fatalf("write source: %v", err)
	}
	var synced []string
	onSync = func(d string) { synced = append(synced, d) }
	t.Cleanup(func() { onSync = nil })

	if err := NewQuarantine(dir).Admit("seg-3", source); err != nil {
		t.Fatalf("Admit: %v", err)
	}
	// The quarantine directory is linked in the spool, then the entry in the
	// quarantine, then the moved file, and only then the source removal.
	want := []string{spool, dir, filepath.Join(dir, "seg-3"), spool}
	if !slices.Equal(synced, want) {
		t.Fatalf("synced directories = %v, want %v", synced, want)
	}
}

func TestRemoveWaitsForAnAdmitInProgress(t *testing.T) {
	spool := t.TempDir()
	dir := filepath.Join(spool, "quarantine")
	if err := os.MkdirAll(dir, 0o750); err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(spool, "request-trace.000003.jsonl.gz")
	if err := os.WriteFile(source, []byte("segment"), 0o640); err != nil {
		t.Fatal(err)
	}
	q := NewQuarantine(dir)
	// Admit stops once the entry exists and before it moves the file in,
	// the moment a removal running beside intake could see it half built.
	paused, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	onSync = func(d string) {
		if d == dir {
			once.Do(func() { close(paused); <-release })
		}
	}
	t.Cleanup(func() { onSync = nil })

	admitted := make(chan error, 1)
	go func() { admitted <- q.Admit("seg-3", source) }()
	<-paused
	removed := make(chan error, 1)
	go func() { removed <- q.Remove("seg-3") }()
	select {
	case err := <-removed:
		close(release)
		<-admitted
		t.Fatalf("Remove returned %v while Admit was still moving files in, want it to wait", err)
	case <-time.After(200 * time.Millisecond):
	}
	close(release)

	if err := <-admitted; err != nil {
		t.Fatalf("Admit: %v, want the move finished", err)
	}
	if err := <-removed; err != nil {
		t.Fatalf("Remove: %v", err)
	}
	for _, path := range []string{source, filepath.Join(dir, "seg-3")} {
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("%s still exists (%v), want the whole entry removed after the admit", path, err)
		}
	}
}
