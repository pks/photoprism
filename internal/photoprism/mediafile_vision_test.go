package photoprism

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/sirupsen/logrus"
	logtest "github.com/sirupsen/logrus/hooks/test"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/photoprism/photoprism/internal/ai/classify"
	"github.com/photoprism/photoprism/internal/ai/nsfw"
	"github.com/photoprism/photoprism/internal/ai/vision"
	"github.com/photoprism/photoprism/internal/config"
	"github.com/photoprism/photoprism/internal/entity"
	"github.com/photoprism/photoprism/internal/event"
	"github.com/photoprism/photoprism/pkg/fs"
	"github.com/photoprism/photoprism/pkg/media"
	"github.com/photoprism/photoprism/pkg/rnd"
)

func setupVisionMediaFile(t *testing.T) *MediaFile {
	t.Helper()

	cfg := config.TestConfig()
	require.NoError(t, cfg.InitializeTestData())

	mediaFile, err := NewMediaFile("testdata/flash.jpg")
	require.NoError(t, err)

	return mediaFile
}

func TestMediaFile_GenerateCaption(t *testing.T) {
	mediaFile := setupVisionMediaFile(t)

	originalConfig := vision.Config
	t.Cleanup(func() {
		vision.Config = originalConfig
		vision.SetCaptionFunc(nil)
	})

	captionModel := &vision.Model{Type: vision.ModelTypeCaption, Engine: vision.ApiFormatOpenAI}
	captionModel.ApplyEngineDefaults()
	vision.Config = &vision.ConfigValues{Models: vision.Models{captionModel}}

	t.Run("AutoUsesModelSource", func(t *testing.T) {
		vision.SetCaptionFunc(func(files vision.Files, mediaSrc media.Src, _ string) (*vision.CaptionResult, *vision.Model, error) {
			return &vision.CaptionResult{Text: "stub", Source: captionModel.GetSource()}, captionModel, nil
		})

		caption, err := mediaFile.GenerateCaption(entity.SrcAuto, "")
		require.NoError(t, err)
		require.NotNil(t, caption)
		assert.Equal(t, captionModel.GetSource(), caption.Source)
	})
	t.Run("CustomSourceOverrides", func(t *testing.T) {
		vision.SetCaptionFunc(func(files vision.Files, mediaSrc media.Src, _ string) (*vision.CaptionResult, *vision.Model, error) {
			return &vision.CaptionResult{Text: "stub", Source: captionModel.GetSource()}, captionModel, nil
		})

		caption, err := mediaFile.GenerateCaption(entity.SrcManual, "")
		require.NoError(t, err)
		require.NotNil(t, caption)
		assert.Equal(t, entity.SrcManual, caption.Source)
	})
	t.Run("ExtraMetadataIsPassedToTheModel", func(t *testing.T) {
		var promptMetadata string
		vision.SetCaptionFunc(func(files vision.Files, mediaSrc media.Src, metadata string) (*vision.CaptionResult, *vision.Model, error) {
			promptMetadata = metadata
			return &vision.CaptionResult{Text: "stub", Source: captionModel.GetSource()}, captionModel, nil
		})

		caption, err := mediaFile.GenerateCaption(entity.SrcAuto, VisionPeoplePart([]string{"Anna", " Ben "}, 0))
		require.NoError(t, err)
		require.NotNil(t, caption)
		assert.Contains(t, promptMetadata, "People Anna, Ben")
	})
	t.Run("MissingModelReturnsError", func(t *testing.T) {
		vision.Config = &vision.ConfigValues{}
		vision.SetCaptionFunc(nil)

		caption, err := mediaFile.GenerateCaption(entity.SrcAuto, "")
		assert.Error(t, err)
		assert.Nil(t, caption)
	})
}

func TestMediaFile_GenerateLabels(t *testing.T) {
	mediaFile := setupVisionMediaFile(t)

	originalConfig := vision.Config
	t.Cleanup(func() {
		vision.Config = originalConfig
		vision.SetLabelsFunc(nil)
	})

	labelModel := &vision.Model{Type: vision.ModelTypeLabels, Engine: vision.ApiFormatOllama}
	labelModel.ApplyEngineDefaults()
	vision.Config = &vision.ConfigValues{Models: vision.Models{labelModel}}

	wantModel, _, _ := labelModel.GetModel()

	t.Run("AutoUsesModelSource", func(t *testing.T) {
		var captured string
		vision.SetLabelsFunc(func(files vision.Files, mediaSrc media.Src, src entity.Src, _ string) (classify.Labels, error) {
			captured = src
			return classify.Labels{{Name: "stub", Source: src}}, nil
		})

		labels, modelName := mediaFile.GenerateLabels(entity.SrcAuto, "")
		assert.NotEmpty(t, labels)
		assert.Equal(t, labelModel.GetSource(), captured)
		assert.Equal(t, wantModel, modelName)
	})
	t.Run("CustomSourceOverrides", func(t *testing.T) {
		var captured string
		vision.SetLabelsFunc(func(files vision.Files, mediaSrc media.Src, src entity.Src, _ string) (classify.Labels, error) {
			captured = src
			return classify.Labels{{Name: "stub", Source: src}}, nil
		})

		labels, _ := mediaFile.GenerateLabels(entity.SrcManual, "")
		assert.NotEmpty(t, labels)
		assert.Equal(t, entity.SrcManual, captured)
	})
	t.Run("MissingModel", func(t *testing.T) {
		vision.Config = &vision.ConfigValues{}
		vision.SetLabelsFunc(nil)

		labels, modelName := mediaFile.GenerateLabels(entity.SrcAuto, "")
		assert.Empty(t, labels)
		assert.Empty(t, modelName)
	})
}

func TestMediaFile_DetectNSFW(t *testing.T) {
	mediaFile := setupVisionMediaFile(t)

	t.Run("FlagsHighConfidence", func(t *testing.T) {
		vision.SetNSFWFunc(func(files vision.Files, mediaSrc media.Src) ([]nsfw.Result, error) {
			return []nsfw.Result{nsfw.NewResult(0.99, nsfw.DefaultThreshold)}, nil
		})
		t.Cleanup(func() { vision.SetNSFWFunc(nil) })

		result := mediaFile.DetectNSFW()
		assert.True(t, result.IsUnsafe())
		assert.False(t, result.IsSafe())
	})
	t.Run("SafeContent", func(t *testing.T) {
		vision.SetNSFWFunc(func(files vision.Files, mediaSrc media.Src) ([]nsfw.Result, error) {
			return []nsfw.Result{nsfw.NewResult(0.02, nsfw.DefaultThreshold)}, nil
		})
		t.Cleanup(func() { vision.SetNSFWFunc(nil) })

		result := mediaFile.DetectNSFW()
		assert.True(t, result.IsSafe())
		assert.False(t, result.IsUnsafe())
	})
	// A detector that could not answer must not report a clearance, because every caller of
	// this method treats "not unsafe" as permission to leave a photo public.
	t.Run("DetectorError", func(t *testing.T) {
		vision.SetNSFWFunc(func(files vision.Files, mediaSrc media.Src) ([]nsfw.Result, error) {
			return nil, errors.New("model is unavailable")
		})
		t.Cleanup(func() { vision.SetNSFWFunc(nil) })

		logHook, systemHook := captureGeneralAndSystemLog(t)
		result := mediaFile.DetectNSFW()
		assert.True(t, result.IsUnavailable())
		assert.False(t, result.IsSafe())
		assert.False(t, result.IsUnsafe())
		for _, entry := range logHook.AllEntries() {
			assert.NotContains(t, entry.Message, "model is unavailable")
		}
		require.Len(t, systemHook.AllEntries(), 1)
		assert.Equal(t, "vision: model is unavailable in testdata/flash.jpg (detect nsfw)", systemHook.AllEntries()[0].Message)
	})
	t.Run("NoResult", func(t *testing.T) {
		vision.SetNSFWFunc(func(files vision.Files, mediaSrc media.Src) ([]nsfw.Result, error) {
			return []nsfw.Result{}, nil
		})
		t.Cleanup(func() { vision.SetNSFWFunc(nil) })

		result := mediaFile.DetectNSFW()
		assert.True(t, result.IsUnavailable())
		assert.False(t, result.IsSafe())
	})
	t.Run("UndecidedResult", func(t *testing.T) {
		vision.SetNSFWFunc(func(files vision.Files, mediaSrc media.Src) ([]nsfw.Result, error) {
			return []nsfw.Result{nsfw.Unavailable("thumbnail is missing")}, nil
		})
		t.Cleanup(func() { vision.SetNSFWFunc(nil) })

		result := mediaFile.DetectNSFW()
		assert.True(t, result.IsUnavailable())
		assert.False(t, result.IsSafe())
	})
}

func TestMediaFile_VisionMetadata(t *testing.T) {
	t.Run("NilReturnsEmpty", func(t *testing.T) {
		var m *MediaFile
		assert.Equal(t, "", m.visionMetadata())
	})
	t.Run("DateCameraAndName", func(t *testing.T) {
		mediaFile, err := NewMediaFile("testdata/flash.jpg")
		require.NoError(t, err)
		assert.Equal(t, "Taken 2013-03-05; Camera Canon EOS 6D; 100mm f/4.5 ISO 500 1/80; Lens EF100mm f/2.8L Macro IS USM; Flash used; Description Animal with green eyes on table burst; Keywords burst, flash, marienkäfer, panorama; File name flash.jpg", mediaFile.visionMetadata())
	})
	t.Run("GpsWithoutPlaceHasNoPlacePart", func(t *testing.T) {
		mediaFile, err := NewMediaFile("testdata/digikam.jpg")
		require.NoError(t, err)
		metadata := mediaFile.visionMetadata()
		assert.Contains(t, metadata, "Taken 2020-10-17")
		assert.Contains(t, metadata, "Camera HUAWEI ELE-L29")
		assert.Contains(t, metadata, "27mm f/1.8 ISO 100 1/50")
		assert.Contains(t, metadata, "Subject Berlin, Shop")
		assert.Contains(t, metadata, "Keywords berlin, shop")
		assert.Contains(t, metadata, "File name digikam.jpg")
		assert.NotContains(t, metadata, "Place ")
	})
	t.Run("NoExifOnlyName", func(t *testing.T) {
		mediaFile, err := NewMediaFile("testdata/animated.png")
		require.NoError(t, err)
		metadata := mediaFile.visionMetadata()
		assert.Contains(t, metadata, "File name animated.png")
		assert.NotContains(t, metadata, "Camera")
		assert.NotContains(t, metadata, "ISO")
	})
	t.Run("VideoDurationAndResolution", func(t *testing.T) {
		mediaFile, err := NewMediaFile("../ffmpeg/testdata/30fps.mov")
		require.NoError(t, err)
		metadata := mediaFile.visionMetadata()
		assert.Contains(t, metadata, "Video 1920x1080")
		assert.Contains(t, metadata, "File name 30fps.mov")
	})
}

func TestVisionPeoplePart(t *testing.T) {
	assert.Equal(t, "", VisionPeoplePart(nil, 0))
	assert.Equal(t, "", VisionPeoplePart([]string{"", "  "}, 0))
	assert.Equal(t, "People Anna, Ben", VisionPeoplePart([]string{"Anna", " Ben "}, 0))
	assert.Equal(t, "People Anna", VisionPeoplePart([]string{"Anna"}, 3))
	assert.Equal(t, "People detected: 3", VisionPeoplePart(nil, 3))
}

func TestVisionPlacePart(t *testing.T) {
	assert.Equal(t, "", VisionPlacePart(nil))
	assert.Equal(t, "", VisionPlacePart(&entity.Photo{}))
	assert.Equal(t, "Place 10.0000, -20.0000", VisionPlacePart(&entity.Photo{PhotoLat: 10, PhotoLng: -20}))

	place := &entity.Place{ID: "test:place", PlaceLabel: "Testville, Testland"}
	photo := &entity.Photo{PhotoLat: 10, PhotoLng: -20, PlaceID: "test:place", Place: place}
	assert.Equal(t, "Place Testville, Testland", VisionPlacePart(photo))
}

func TestVisionPhotoPart(t *testing.T) {
	t.Run("PeopleAndKnownLabelsOnly", func(t *testing.T) {
		// A photo without a UID skips the album lookup, so no database is needed.
		photo := &entity.Photo{
			Labels: entity.PhotoLabels{
				{Label: &entity.Label{LabelName: "dog"}, LabelSrc: entity.SrcManual},
				{Label: &entity.Label{LabelName: "llm"}, LabelSrc: entity.SrcOpenAI},
				{Label: &entity.Label{LabelName: "image"}, LabelSrc: entity.SrcImage},
			},
		}

		part := VisionPhotoPart(photo, 2, nil)

		assert.Contains(t, part, "People detected: 2")
		assert.Contains(t, part, "Known labels: dog")
		assert.NotContains(t, part, "llm")
		assert.NotContains(t, part, "image")
	})
	t.Run("KnownLabelsCappedAtTwenty", func(t *testing.T) {
		// Uncertainty decreases with i, so label-24 is most confident and the label-00..04
		// tail must be dropped by the cap.
		var labels entity.PhotoLabels

		for i := 0; i < 25; i++ {
			labels = append(labels, entity.PhotoLabel{
				Label:       &entity.Label{LabelName: fmt.Sprintf("label-%02d", i)},
				LabelSrc:    entity.SrcMeta,
				Uncertainty: 90 - i,
			})
		}

		photo := &entity.Photo{Labels: labels}
		part := VisionPhotoPart(photo, 0, nil)

		assert.Contains(t, part, "label-24")
		assert.Contains(t, part, "label-05")
		assert.NotContains(t, part, "label-04")
	})
	t.Run("PlacePartFromPhotoLocation", func(t *testing.T) {
		// The photo-level location is used, so a photo with coordinates gets a place part
		// even though the album lookup is skipped without a UID.
		photo := &entity.Photo{PhotoLat: 10, PhotoLng: -20}

		part := VisionPhotoPart(photo, 0, nil)

		assert.Contains(t, part, "Place 10.0000, -20.0000")
	})
	t.Run("AlbumsAreIncluded", func(t *testing.T) {
		cfg := config.TestConfig()
		require.NoError(t, cfg.InitializeTestData())

		data, err := os.ReadFile(filepath.Join("testdata", "flash.jpg"))
		require.NoError(t, err)

		// A random name and trailing bytes make each file a new photo, also in repeated runs.
		relName := "vision-photo-part-" + rnd.Base36(8) + ".jpg"
		fileName := filepath.Join(cfg.OriginalsPath(), relName)
		require.NoError(t, os.MkdirAll(filepath.Dir(fileName), fs.ModeDir))
		require.NoError(t, os.WriteFile(fileName, append(append([]byte{}, data...), []byte(relName)...), fs.ModeFile)) //nolint:gosec // isolated test path
		t.Cleanup(func() {
			_ = os.Remove(fileName)
			_ = os.Remove(filepath.Dir(fileName))
		})

		mediaFile, fileErr := NewMediaFile(fileName)
		require.NoError(t, fileErr)

		ind := NewIndex(cfg, NewConvert(cfg), NewFiles(), NewPhotos())
		result := ind.UserMediaFile(mediaFile, NewIndexOptions("/", false, false, false, false, true, cfg), relName, "", entity.OwnerUnknown)
		require.Equal(t, IndexAdded, result.Status, "%s", result.Err)

		photo := entity.FindPhoto(entity.Photo{PhotoUID: result.PhotoUID})
		require.NotNil(t, photo)
		t.Cleanup(func() { _, _ = photo.DeletePermanently() })

		require.NoError(t, entity.AddPhotoToAlbums(photo.PhotoUID, []string{"Italy Trip"}))
		t.Cleanup(func() {
			_ = entity.UnscopedDb().Where("photo_uid = ?", photo.PhotoUID).Delete(&entity.PhotoAlbum{}).Error
			_ = entity.UnscopedDb().Where("album_title = ?", "Italy Trip").Delete(&entity.Album{}).Error
		})

		part := VisionPhotoPart(photo, 0, nil)
		assert.Contains(t, part, "Albums: Italy Trip")
	})
}

// captureGeneralAndSystemLog replaces the package and system loggers for the duration of a test.
func captureGeneralAndSystemLog(t *testing.T) (logHook, systemHook *logtest.Hook) {
	t.Helper()

	logger, logHook := logtest.NewNullLogger()
	logger.SetLevel(logrus.TraceLevel)
	systemLogger, systemHook := logtest.NewNullLogger()
	systemLogger.SetLevel(logrus.TraceLevel)
	prevLog, prevSystem := log, event.SystemLog
	t.Cleanup(func() { log, event.SystemLog = prevLog, prevSystem })
	log = logger
	event.SystemLog = systemLogger

	return logHook, systemHook
}
