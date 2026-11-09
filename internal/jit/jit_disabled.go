//go:build !quickjs_jit || android || ios || !((linux && amd64) || (windows && amd64) || (darwin && arm64) || (linux && arm64))

package jit

func executablePolicy() error { return ErrUnavailable }

func mapCode(int) ([]byte, error) { return nil, ErrUnavailable }

func protectCode([]byte, bool) error { return ErrUnavailable }

func flushCode([]byte) error { return ErrUnavailable }

func unmapCode([]byte) error { return nil }
