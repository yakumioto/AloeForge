package clientruntime

import (
	"encoding/json"
	"io"
	"strings"
	"testing"
	"unicode/utf8"

	p "github.com/yakumioto/AloeForge/internal/runtimeprotocol"
)

func TestUTF8AndStopPrefixes(t *testing.T) {
	d := textDecoder{stops: []string{"STOP", "中文"}}
	var output strings.Builder
	for _, piece := range [][]byte{[]byte("aST"), []byte("x"), {0xe4}, {0xb8}, {0xad}, []byte("文trailing")} {
		text, stopped, err := d.add(piece)
		if err != nil || !utf8.ValidString(text) {
			t.Fatal("decoder rejected valid pieces")
		}
		output.WriteString(text)
		if stopped {
			break
		}
		if len(d.utf8Tail) > 3 || len(d.pending) > MaxStopBytes {
			t.Fatal("unbounded decoder carry")
		}
	}
	if output.String() != "aSTx" {
		t.Fatal("stop prefix leaked or ordinary prefix was lost", output.String())
	}
	for _, piece := range [][]byte{{0xff}, {0xc0, 0x80}, {0xed, 0xa0, 0x80}} {
		d := textDecoder{}
		if _, _, err := d.add(piece); err == nil {
			t.Fatal("invalid UTF-8 accepted")
		}
	}
}
func FuzzSSE(f *testing.F) {
	event := p.Event{RequestID: strings.Repeat("a", 32), Sequence: 1, OutputTokenIDs: []int{3}, FinishReason: "", Error: ""}
	data, _ := json.Marshal(event)
	f.Add([]byte("event: token\ndata: " + string(data) + "\n\n"))
	f.Add([]byte("data: {}\n\n"))
	f.Add([]byte("event: token\ndata: " + strings.Repeat("x", p.MaxEvent+1) + "\n\n"))
	f.Fuzz(func(t *testing.T, input []byte) {
		if len(input) > 2*p.MaxEvent {
			return
		}
		reader := newEvents(strings.NewReader(string(input)))
		// Bounded iteration, independently of bytes containing many tiny records.
		for i := 0; i < 4; i++ {
			got, err := reader.next()
			if err != nil {
				return
			}
			if got.Validate(event.RequestID, i+1, 16) == nil {
				encoded, err := json.Marshal(got)
				if err != nil || len(encoded) > p.MaxEvent {
					t.Fatal("accepted oversized record")
				}
			}
		}
	})
}
func FuzzUTF8Pieces(f *testing.F) {
	f.Add([]byte("A中文STx!"), uint8(1))
	f.Add([]byte("STOPtail"), uint8(3))
	f.Add([]byte{0xe4, 0xb8, 0xad}, uint8(2))
	f.Add([]byte{0xff}, uint8(1))
	f.Fuzz(func(t *testing.T, data []byte, width uint8) {
		if len(data) > 4096 {
			return
		}
		step := int(width)%32 + 1
		d := textDecoder{stops: []string{"STOP"}}
		var result strings.Builder
		stopped := false
		for offset := 0; offset < len(data); offset += step {
			end := min(offset+step, len(data))
			text, done, err := d.add(data[offset:end])
			if err != nil {
				if utf8.Valid(data) {
					t.Fatal("valid UTF-8 rejected while decoding")
				}
				return
			}
			if !utf8.ValidString(text) || len(d.utf8Tail) > 3 || len(d.pending) >= len("STOP") {
				t.Fatal("decoder violated UTF-8/carry bound")
			}
			result.WriteString(text)
			if done {
				stopped = true
				break
			}
		}
		if !stopped {
			tail, err := d.finish()
			if err != nil {
				if utf8.Valid(data) {
					t.Fatal("valid UTF-8 rejected at finish")
				}
				return
			}
			result.WriteString(tail)
		}
		// Independent whole-string oracle only when the complete input is valid.
		if utf8.Valid(data) {
			want := string(data)
			if i := strings.Index(want, "STOP"); i >= 0 {
				want = want[:i]
			}
			if result.String() != want {
				t.Fatal("incremental decoder differs from complete UTF-8 string")
			}
		}
	})
}

// The parser must not require an entire unbounded stream in memory.
type repeatingReader struct{}

func (repeatingReader) Read(b []byte) (int, error) {
	for i := range b {
		b[i] = 'x'
	}
	return len(b), nil
}
func TestInfiniteLineStopsAtBound(t *testing.T) {
	reader := newEvents(io.Reader(repeatingReader{}))
	if _, err := reader.next(); err == nil {
		t.Fatal("unbounded line accepted")
	}
}
