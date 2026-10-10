package clientruntime

import (
	"bufio"
	"errors"
	"io"
	"strings"
	"unicode/utf8"

	p "github.com/yakumioto/AloeForge/internal/runtimeprotocol"
)

type events struct{ scanner *bufio.Scanner }

func newEvents(r io.Reader) *events {
	scanner := bufio.NewScanner(io.LimitReader(r, p.MaxStream+1))
	scanner.Buffer(make([]byte, p.MaxEvent), p.MaxEvent)
	return &events{scanner}
}
func (e *events) next() (p.Event, error) {
	var result p.Event
	if !e.scanner.Scan() || e.scanner.Text() != "event: token" {
		return result, errors.New("invalid SSE record")
	}
	if !e.scanner.Scan() || !strings.HasPrefix(e.scanner.Text(), "data: ") {
		return result, errors.New("invalid SSE record")
	}
	data := []byte(strings.TrimPrefix(e.scanner.Text(), "data: "))
	if p.DecodeEvent(data, &result) != nil {
		return result, errors.New("invalid SSE payload")
	}
	if !e.scanner.Scan() || e.scanner.Text() != "" {
		return result, errors.New("invalid SSE boundary")
	}
	return result, nil
}

// textDecoder holds at most three incomplete UTF-8 bytes and a suffix that
// could still become a stop string. It never emits replacement runes or leaks
// a stop-string prefix to callbacks. Tokens are not re-encoded.
type textDecoder struct {
	utf8Tail []byte
	pending  string
	stops    []string
}

func (d *textDecoder) add(piece []byte) (string, bool, error) {
	data := append(d.utf8Tail, piece...)
	end := 0
	for end < len(data) {
		if !utf8.FullRune(data[end:]) {
			break
		}
		r, n := utf8.DecodeRune(data[end:])
		if r == utf8.RuneError && n == 1 {
			return "", false, errors.New("invalid UTF-8")
		}
		end += n
	}
	d.utf8Tail = append([]byte(nil), data[end:]...)
	text := d.pending + string(data[:end])
	d.pending = ""
	first := -1
	for _, stop := range d.stops {
		i := strings.Index(text, stop)
		if i >= 0 && (first < 0 || i < first) {
			first = i
		}
	}
	if first >= 0 {
		d.utf8Tail = nil
		return text[:first], true, nil
	}
	keep := 0
	for _, stop := range d.stops {
		for n := 1; n < len(stop) && n <= len(text); n++ {
			if n > keep && strings.HasSuffix(text, stop[:n]) {
				keep = n
			}
		}
	}
	d.pending = text[len(text)-keep:]
	return text[:len(text)-keep], false, nil
}
func (d *textDecoder) finish() (string, error) {
	if len(d.utf8Tail) != 0 {
		return "", errors.New("truncated UTF-8")
	}
	text := d.pending
	d.pending = ""
	return text, nil
}
