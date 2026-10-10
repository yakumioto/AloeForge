package strictjson

import (
	"strings"
	"testing"
)

func TestRejectInvalidJSON(t *testing.T) {
	for _, input := range []string{
		`{"x":1,"x":2}`, `{"x":{"y":1,"y":2}}`, `{"x":1,"\u0078":2}`,
		`{"x":1} {}`, `{"x":}`, `{"x":NaN}`, `{"x":"\ud800"}`, `{"x":"\udc00"}`,
		"{\"x\":\"\xff\"}", strings.Repeat("[", MaxDepth+2) + "0" + strings.Repeat("]", MaxDepth+2),
	} {
		t.Run(input, func(t *testing.T) {
			var value any
			if Decode([]byte(input), &value, false) == nil {
				t.Fatal("invalid JSON accepted")
			}
		})
	}
}

func TestValidAndUnknownFields(t *testing.T) {
	for _, input := range []string{`{"x":"\ud83d\ude00"}`, `{"x":"\\ud800"}`, `{"x":[]}`} {
		var value any
		if err := Decode([]byte(input), &value, false); err != nil {
			t.Fatal(err)
		}
	}
	var value struct {
		X int `json:"x"`
	}
	if Decode([]byte(`{"x":1,"y":2}`), &value, true) == nil {
		t.Fatal("unknown field accepted")
	}
}
