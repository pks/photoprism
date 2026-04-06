package vision

import (
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/photoprism/photoprism/internal/ai/vision/ollama"
	"github.com/photoprism/photoprism/pkg/http/scheme"
	"github.com/photoprism/photoprism/pkg/media"
)

func TestGenerateCaption(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	} else if _, err := net.DialTimeout("tcp", "photoprism-vision:5000", 10*time.Second); err != nil {
		t.Skip("skipping test because photoprism-vision is not running.")
	}

	t.Run("Success", func(t *testing.T) {
		expectedText := "An image of sound waves"

		result, model, err := GenerateCaption(Files{"https://dl.photoprism.app/img/artwork/colorwaves-400.jpg"}, media.SrcRemote)

		assert.NoError(t, err)
		assert.NotNil(t, model)
		assert.IsType(t, CaptionResult{}, result)
		assert.LessOrEqual(t, float32(0.0), result.Confidence)

		t.Logf("caption: %#v", result)

		assert.Equal(t, expectedText, result.Text)
	})
	t.Run("Invalid", func(t *testing.T) {
		result, model, err := GenerateCaption(nil, media.SrcLocal)

		assert.Error(t, err)
		assert.Nil(t, model)
		assert.IsType(t, CaptionResult{}, result)
		assert.Equal(t, "", result.Text)
		assert.Equal(t, float32(0.0), result.Confidence)
	})
}

func TestCaptionRetryWithoutThinking(t *testing.T) {
	prevConfig := Config
	t.Cleanup(func() { Config = prevConfig })

	t.Run("RetriesWhenThinkingExhaustedBudget", func(t *testing.T) {
		requestCount := 0

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			requestCount++

			var body map[string]any
			require.NoError(t, json.NewDecoder(r.Body).Decode(&body))

			if requestCount == 1 {
				// Simulate thinking exhausting the token budget: response is empty.
				thinkVal, _ := body["think"].(bool)
				assert.True(t, thinkVal, "first request should have think:true")
				require.NoError(t, json.NewEncoder(w).Encode(ollama.Response{
					Model:    "gemma4:26b",
					Response: "",
				}))
			} else {
				// Retry with think:false should return a real caption.
				thinkVal, _ := body["think"].(bool)
				assert.False(t, thinkVal, "retry request should have think:false")
				require.NoError(t, json.NewEncoder(w).Encode(ollama.Response{
					Model:    "gemma4:26b",
					Response: "A tabby cat rests on a sun-lit windowsill.",
				}))
			}
		}))
		defer server.Close()

		model := &Model{
			Type:   ModelTypeCaption,
			Engine: ollama.EngineName,
			Service: Service{
				Uri:            server.URL,
				Method:         http.MethodPost,
				RequestFormat:  ApiFormatOllama,
				ResponseFormat: ApiFormatOllama,
				FileScheme:     scheme.Base64,
				Think:          "true",
			},
		}
		model.ApplyEngineDefaults()

		Config = &ConfigValues{
			Models:     Models{model},
			Thresholds: DefaultThresholds,
		}

		caption, _, err := GenerateCaption(Files{samplesPath + "/cat_224.jpeg"}, media.SrcLocal)
		require.NoError(t, err)
		require.NotNil(t, caption)
		assert.Equal(t, "A tabby cat rests on a sun-lit windowsill.", caption.Text)
		assert.Equal(t, 2, requestCount, "expected two requests: initial with think:true + retry with think:false")
	})

	t.Run("NoRetryWhenThinkNotSet", func(t *testing.T) {
		requestCount := 0

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			requestCount++
			require.NoError(t, json.NewEncoder(w).Encode(ollama.Response{
				Model:    "gemma4:26b",
				Response: "",
			}))
		}))
		defer server.Close()

		model := &Model{
			Type:   ModelTypeCaption,
			Engine: ollama.EngineName,
			Service: Service{
				Uri:            server.URL,
				Method:         http.MethodPost,
				RequestFormat:  ApiFormatOllama,
				ResponseFormat: ApiFormatOllama,
				FileScheme:     scheme.Base64,
			},
		}
		model.ApplyEngineDefaults()

		Config = &ConfigValues{
			Models:     Models{model},
			Thresholds: DefaultThresholds,
		}

		_, _, err := GenerateCaption(Files{samplesPath + "/cat_224.jpeg"}, media.SrcLocal)
		assert.Error(t, err, "expected error when caption is nil and no retry possible")
		assert.Equal(t, 1, requestCount, "expected only one request when Think is not set")
	})
}
