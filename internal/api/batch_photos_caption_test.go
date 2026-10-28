package api

import (
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/photoprism/photoprism/internal/ai/vision"
	"github.com/photoprism/photoprism/internal/config"
	"github.com/photoprism/photoprism/internal/entity"
	"github.com/photoprism/photoprism/internal/mutex"
	"github.com/photoprism/photoprism/internal/photoprism"
	"github.com/photoprism/photoprism/pkg/fs"
	"github.com/photoprism/photoprism/pkg/media"
	"github.com/photoprism/photoprism/pkg/rnd"
)

// stubCaptionModel configures a manual-run caption model with a stubbed result and restores the previous state.
func stubCaptionModel(t *testing.T) *vision.Model {
	t.Helper()

	previous := vision.Config
	t.Cleanup(func() {
		vision.Config = previous
		vision.SetCaptionFunc(nil)
	})

	model := &vision.Model{Type: vision.ModelTypeCaption, Engine: vision.ApiFormatOllama, Run: vision.RunManual}
	model.ApplyEngineDefaults()
	vision.Config = &vision.ConfigValues{Models: vision.Models{model}}

	vision.SetCaptionFunc(func(vision.Files, media.Src, string) (*vision.CaptionResult, *vision.Model, error) {
		return &vision.CaptionResult{Text: "A generated caption", Source: model.GetSource(), Model: "stub-caption-model"}, model, nil
	})

	return model
}

// indexTestPhoto writes a copy of a test fixture into the originals directory and indexes it,
// returning the new photo UID so the vision worker can open the image.
func indexTestPhoto(t *testing.T, cfg *config.Config) string {
	t.Helper()

	data, err := os.ReadFile(filepath.Join("..", "photoprism", "testdata", "flash.jpg"))
	require.NoError(t, err)

	// A random name and trailing bytes make each file a new photo, also in repeated runs.
	relName := filepath.Join("batch-caption", rnd.Base36(8)+".jpg")
	fileName := filepath.Join(cfg.OriginalsPath(), relName)
	require.NoError(t, os.MkdirAll(filepath.Dir(fileName), fs.ModeDir))
	require.NoError(t, os.WriteFile(fileName, append(append([]byte{}, data...), []byte(relName)...), fs.ModeFile)) //nolint:gosec // isolated test path
	t.Cleanup(func() {
		_ = os.Remove(fileName)
		_ = os.Remove(filepath.Dir(fileName))
	})

	mediaFile, err := photoprism.NewMediaFile(fileName)
	require.NoError(t, err)

	opt := photoprism.NewIndexOptions("/", false, false, false, false, true, cfg)
	ind := photoprism.NewIndex(cfg, photoprism.NewConvert(cfg), photoprism.NewFiles(), photoprism.NewPhotos())
	result := ind.UserMediaFile(mediaFile, opt, relName, "", entity.OwnerUnknown)
	require.Equal(t, photoprism.IndexAdded, result.Status, "%s", result.Err)

	return result.PhotoUID
}

func TestBatchPhotosCaption(t *testing.T) {
	t.Run("NoPhotos", func(t *testing.T) {
		app, router, _ := NewApiTest()
		stubCaptionModel(t)
		BatchPhotosCaption(router)

		resp := PerformRequestWithBody(app, "POST", "/api/v1/batch/photos/caption", `{"photos": []}`)

		assert.Equal(t, http.StatusBadRequest, resp.Code)
	})
	t.Run("NoModelConfigured", func(t *testing.T) {
		app, router, _ := NewApiTest()

		previous := vision.Config
		vision.Config = &vision.ConfigValues{}
		t.Cleanup(func() { vision.Config = previous })

		BatchPhotosCaption(router)

		resp := PerformRequestWithBody(app, "POST", "/api/v1/batch/photos/caption", `{"photos": ["pqkm36fjqvset9uy"]}`)

		assert.Equal(t, http.StatusPreconditionFailed, resp.Code)
		assert.Contains(t, resp.Body.String(), "No caption model configured")
	})
	t.Run("Busy", func(t *testing.T) {
		app, router, _ := NewApiTest()
		stubCaptionModel(t)

		require.NoError(t, mutex.VisionWorker.Start())
		t.Cleanup(func() { mutex.VisionWorker.Stop() })

		BatchPhotosCaption(router)

		resp := PerformRequestWithBody(app, "POST", "/api/v1/batch/photos/caption", `{"photos": ["pqkm36fjqvset9uy"]}`)

		assert.Equal(t, http.StatusTooManyRequests, resp.Code)
	})
	t.Run("Success", func(t *testing.T) {
		app, router, cfg := NewApiTest()
		stubCaptionModel(t)
		BatchPhotosCaption(router)

		photoUID := indexTestPhoto(t, cfg)
		t.Cleanup(func() {
			if photo := entity.FindPhoto(entity.Photo{PhotoUID: photoUID}); photo != nil {
				_, _ = photo.DeletePermanently()
			}
		})

		// The fixture file carries an EXIF caption with higher source priority, so force is required.
		resp := PerformRequestWithBody(app, "POST", "/api/v1/batch/photos/caption", `{"photos": ["`+photoUID+`"], "force": true}`)
		assert.Equal(t, http.StatusAccepted, resp.Code)

		// The worker runs in the background; wait for the caption to be stored.
		deadline := time.Now().Add(30 * time.Second)
		for time.Now().Before(deadline) {
			photo := entity.FindPhoto(entity.Photo{PhotoUID: photoUID})
			if photo != nil && photo.GetCaption() == "A generated caption" {
				break
			}
			time.Sleep(200 * time.Millisecond)
		}

		photo := entity.FindPhoto(entity.Photo{PhotoUID: photoUID})
		require.NotNil(t, photo)
		assert.Equal(t, "A generated caption", photo.GetCaption())
		assert.Equal(t, entity.SrcOllama, photo.GetCaptionSrc())
		assert.Equal(t, "stub-caption-model", photo.GetDetails().CaptionModel)
	})
}
