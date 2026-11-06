//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"weak"

	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
)

// A literal keeps the capacity the tree tier confirmed at a turn's end
// before it reached native code. The program counter, unlike the tree
// node's ordinal, also identifies the literal in an inlined callee.
func jitLiteralArraySource[_ any](r *Runtime, cl *closure, pool *abi.ObjectPool, pc, n int) *jitPoolSource {
	s := &jitPoolSource{array: true, n: n}
	for _, a := range cl.arrays {
		if a != nil && a.acrossTurns && a.pc == int32(pc) && a.room >= 256 && a.room <= arrayRoomLimit {
			s.room = a.room
			if r.jit != nil {
				r.jit.arrayRooms = append(r.jit.arrayRooms, weak.Make(pool))
			}
			break
		}
	}
	return s
}

// A pool with a large hint holds one array. A batch would retain unused
// buffers and delay observing a changed length. Native code may shrink a
// hint, but only the tree tier's two samples may establish a large one.
func jitLiteralArrayRoom[_ any](pool *abi.ObjectPool, n int) int {
	src := (*jitPoolSource)(pool.Source)
	if src == nil {
		return n
	}
	if src.room >= 256 {
		if a := src.sample; a != nil {
			length := len(a.elemsOf())
			if length > arrayRoomLimit {
				length = jitLiteralRoom
			}
			src.room = min(src.room, length)
			if src.room < 256 {
				src.room = min(src.room, jitLiteralRoom)
			}
		}
		// Do not retain the previous result's graph through a collection
		// triggered by allocating its replacement's element buffer.
		src.sample = nil
		if src.room >= 256 {
			pool.Size = 1
		}
	}
	return max(n, src.room)
}

// A one-array pool's sample is its completed result. Observe and release
// it at turn end, so the hint cannot retain a page between requests. An
// unused pool still owns its empty array and has no result to observe.
func jitLiteralArraysLetGo[_ any](r *Runtime) {
	s := r.jit
	if s == nil || len(s.arrayRooms) == 0 || r.frameDepth != 0 {
		return
	}
	jitLiteralArraySweep[struct{}](s)
}

func jitLiteralArraySweep[_ any](s *jitState) {
	keep := s.arrayRooms[:0]
	for _, w := range s.arrayRooms {
		pool := w.Value()
		if pool == nil {
			continue
		}
		src := (*jitPoolSource)(pool.Source)
		if src == nil || src.room < 256 {
			continue
		}
		if pool.Count == 0 {
			jitLiteralArrayRoom[struct{}](pool, src.n)
		}
		if src.room >= 256 {
			keep = append(keep, w)
		}
	}
	clear(s.arrayRooms[len(keep):])
	s.arrayRooms = keep
}
