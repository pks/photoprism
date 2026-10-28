package photoprism

import (
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/dustin/go-humanize/english"

	"github.com/photoprism/photoprism/internal/ai/classify"
	"github.com/photoprism/photoprism/internal/ai/nsfw"
	"github.com/photoprism/photoprism/internal/ai/vision"
	"github.com/photoprism/photoprism/internal/entity"
	"github.com/photoprism/photoprism/internal/event"
	"github.com/photoprism/photoprism/internal/thumb"
	"github.com/photoprism/photoprism/pkg/clean"
	"github.com/photoprism/photoprism/pkg/media"
)

// GenerateCaption generates a caption for the provided media file using the active
// vision model. When captionSrc is SrcAuto the model's declared source is used;
// otherwise the explicit source is recorded on the returned caption.
// extra may hold photo-level context (people, known labels, albums) appended to the model prompt.
func (m *MediaFile) GenerateCaption(captionSrc entity.Src, extra string) (caption *vision.CaptionResult, err error) {
	start := time.Now()

	model := vision.Config.Model(vision.ModelTypeCaption)

	// No caption generation model configured or usable.
	if model == nil {
		return caption, errors.New("no caption model configured")
	}

	if captionSrc == entity.SrcAuto {
		captionSrc = model.GetSource()
	}

	size := vision.Thumb(vision.ModelTypeCaption)

	// Get thumbnail filenames for the selected sizes.
	fileName, fileErr := m.Thumbnail(Config().ThumbCachePath(), size.Name)

	if fileErr != nil {
		return caption, fileErr
	}

	metadata := appendMetadata(m.visionMetadata(), extra)

	// Get matching labels from computer vision model.
	// Generate a caption using the configured vision model.
	if caption, _, err = vision.GenerateCaption(vision.Files{fileName}, media.SrcLocal, metadata); err != nil {
		// Failed.
	} else if caption.Text != "" {
		if captionSrc != entity.SrcAuto {
			caption.Source = captionSrc
		}

		log.Infof("vision: generated caption for %s [%s]", clean.Log(m.RootRelName()), time.Since(start))
	}

	return caption, err
}

// GenerateLabels classifies the media file and returns matching labels. When labelSrc
// is SrcAuto the model's declared source is used; otherwise the provided source
// is applied to every returned label.
// extra may hold photo-level context (people, known labels, albums) appended to the model prompt.
// The second return value is the name of the model that produced the labels, or "" when none ran.
func (m *MediaFile) GenerateLabels(labelSrc entity.Src, extra string) (labels classify.Labels, modelName string) {
	if m == nil {
		return labels, modelName
	}

	start := time.Now()

	var err error
	var sizes []thumb.Name
	var thumbnails []string

	model := vision.Config.Model(vision.ModelTypeLabels)

	// No label generation model configured or usable.
	if model == nil {
		return labels, modelName
	}

	modelName, _, _ = model.GetModel()

	if labelSrc == entity.SrcAuto {
		labelSrc = model.GetSource()
	}

	if model.UsesPreparedLabels() {
		inputs, inputErr := m.PrepareLabelInputs()
		if inputErr != nil {
			log.Debugf("labels: %s in %s", inputErr, clean.Log(m.RootRelName()))
			return labels, modelName
		}
		labels, err = vision.GeneratePreparedLabels(inputs, labelSrc)
	} else {
		size := vision.Thumb(vision.ModelTypeLabels)

		// The thumbnail size may need to be adjusted to use other models.
		switch {
		case size.Name != "" && size.Name != thumb.Tile224:
			sizes = []thumb.Name{size.Name}
			thumbnails = make([]string, 0, 1)
		case m.Square():
			// Only one thumbnail is required for square images.
			sizes = []thumb.Name{thumb.Tile224}
			thumbnails = make([]string, 0, 1)
		default:
			// Use three thumbnails otherwise (center, left, right).
			sizes = []thumb.Name{thumb.Tile224, thumb.Left224, thumb.Right224}
			thumbnails = make([]string, 0, 3)
		}

		// Get thumbnail filenames for the selected sizes.
		for _, s := range sizes {
			if thumbnail, fileErr := m.Thumbnail(Config().ThumbCachePath(), s); fileErr != nil {
				log.Debugf("%s in %s", fileErr, clean.Log(m.RootRelName()))
				continue
			} else {
				thumbnails = append(thumbnails, thumbnail)
			}
		}

		// Run the configured vision model to obtain labels for the generated thumbnails.
		labels, err = vision.GenerateLabels(thumbnails, media.SrcLocal, labelSrc, appendMetadata(m.visionMetadata(), extra))
	}
	if err != nil {
		log.Debugf("labels: %s in %s", clean.ErrorFull(err), clean.Log(m.RootRelName()))
		return labels, modelName
	}

	// Log number and names of generated labels.
	if n := labels.Count(); n > 0 {
		log.Debugf("vision: %#v", labels)
		log.Infof("vision: generated %s for %s [%s]", english.Plural(n, "label", "labels"), clean.Log(m.RootRelName()), time.Since(start))
	}

	return labels, modelName
}

// visionMetadata returns a compact, data-only summary of the known capture date,
// camera, and file name for use in LLM prompts.
func (m *MediaFile) visionMetadata() string {
	if m == nil {
		return ""
	}

	var parts []string

	if takenAt, _, _ := m.TakenAt(); !takenAt.IsZero() {
		parts = append(parts, "Taken "+takenAt.UTC().Format("2006-01-02"))
	}

	switch {
	case m.CameraMake() != "" && m.CameraModel() != "" && strings.Contains(strings.ToLower(m.CameraModel()), strings.ToLower(m.CameraMake())):
		// Avoid redundant output such as "Canon Canon EOS 6D".
		parts = append(parts, "Camera "+m.CameraModel())
	default:
		if camera := strings.TrimSpace(m.CameraMake() + " " + m.CameraModel()); camera != "" {
			parts = append(parts, "Camera "+camera)
		}
	}

	var spec []string

	if focal := m.FocalLength(); focal > 0 {
		spec = append(spec, fmt.Sprintf("%dmm", focal))
	}

	if f := m.FNumber(); f > 0 {
		spec = append(spec, fmt.Sprintf("f/%.1f", f))
	}

	if iso := m.Iso(); iso > 0 {
		spec = append(spec, "ISO "+strconv.Itoa(iso))
	}

	if exposure := strings.TrimSpace(m.Exposure()); exposure != "" {
		spec = append(spec, exposure)
	}

	if len(spec) > 0 {
		parts = append(parts, strings.Join(spec, " "))
	}

	if lens := strings.TrimSpace(oneLine(m.LensModel())); lens != "" {
		parts = append(parts, "Lens "+lens)
	}

	data := m.MetaData()

	if data.Flash {
		parts = append(parts, "Flash used")
	}

	if info := m.VideoInfo(); info.Duration > 0 || len(info.TrackSizes) > 0 {
		var video []string

		if seconds := int(info.Duration.Seconds()); seconds > 0 {
			video = append(video, fmt.Sprintf("%ds", seconds))
		}

		width, height := 0, 0

		if info.VideoWidth > 0 && info.VideoHeight > 0 {
			width, height = info.VideoWidth, info.VideoHeight
		} else if len(info.TrackSizes) > 0 {
			// Track headers carry the frame size of every visual track, whatever its codec.
			width, height = info.TrackSizes[0].Width, info.TrackSizes[0].Height
		}

		if width > 0 && height > 0 {
			video = append(video, fmt.Sprintf("%dx%d", width, height))
		}

		if len(video) > 0 {
			parts = append(parts, "Video "+strings.Join(video, " "))
		}
	}

	// Existing text metadata is high-signal context, so include it as data only.
	if description := strings.TrimSpace(oneLine(data.Caption)); description != "" {
		parts = append(parts, "Description "+description)
	}

	if subject := strings.TrimSpace(oneLine(data.Subject)); subject != "" {
		parts = append(parts, "Subject "+subject)
	}

	keywords := data.Keywords[:min(len(data.Keywords), 8)]

	if len(keywords) > 0 {
		parts = append(parts, "Keywords "+strings.Join(keywords, ", "))
	}

	// Newlines in file names would break the prompt layout, so collapse them.
	if baseName := strings.TrimSpace(oneLine(m.BaseName())); baseName != "" {
		parts = append(parts, "File name "+baseName)
	}

	return strings.Join(parts, "; ")
}

// cleanNames trims subject names and drops empty values so they are safe to prompt with.
func cleanNames(names []string) []string {
	result := make([]string, 0, len(names))

	for _, name := range names {
		if name = strings.TrimSpace(oneLine(name)); name != "" {
			result = append(result, name)
		}
	}

	return result
}

// oneLine collapses newlines so user-influenced values cannot break the prompt layout.
func oneLine(s string) string {
	return strings.Map(func(r rune) rune {
		if r == '\n' || r == '\r' {
			return ' '
		}
		return r
	}, s)
}

// VisionPeoplePart returns a metadata part listing the known people in the image.
// Without names it reports the detected face count instead. Empty when nothing is known.
func VisionPeoplePart(names []string, faceCount int) string {
	if names := cleanNames(names); len(names) > 0 {
		return "People " + strings.Join(names, ", ")
	}

	if faceCount > 0 {
		return fmt.Sprintf("People detected: %d", faceCount)
	}

	return ""
}

// VisionPhotoPart returns photo-level metadata parts (people, known labels, albums)
// for use in vision model prompts.
// Model-generated labels are excluded so the model is not shown its own previous output.
func VisionPhotoPart(photo *entity.Photo, faceCount int, names []string) string {
	var parts []string

	if part := VisionPeoplePart(names, faceCount); part != "" {
		parts = append(parts, part)
	}

	if part := VisionPlacePart(photo); part != "" {
		parts = append(parts, part)
	}

	if labels := knownLabelNames(photo); len(labels) > 0 {
		parts = append(parts, "Known labels: "+strings.Join(labels, ", "))
	}

	if albums := albumNamesForPhoto(photo); len(albums) > 0 {
		parts = append(parts, "Albums: "+strings.Join(albums, ", "))
	}

	return strings.Join(parts, "; ")
}

// maxVisionLabels limits the known labels passed to vision model prompts so long
// label lists cannot bloat the request.
const maxVisionLabels = 20

// VisionPlacePart returns the photo place as prompt context. It prefers the reverse-geocoded
// place label and falls back to raw coordinates. The photo-level location is used instead of
// file EXIF because the primary file of a photo may lack GPS data another file carries.
func VisionPlacePart(photo *entity.Photo) string {
	if photo == nil || photo.NoLatLng() {
		return ""
	}

	if photo.HasPlace() {
		if err := photo.LoadPlace(); err == nil && photo.Place != nil {
			if label := strings.TrimSpace(photo.Place.Label()); label != "" && label != entity.UnknownPlace.Label() {
				return "Place " + label
			}
		}
	}

	return fmt.Sprintf("Place %.4f, %.4f", photo.PhotoLat, photo.PhotoLng)
}

// knownLabelNames returns the names of the most confident non-model-generated labels on the photo,
// capped at maxVisionLabels.
func knownLabelNames(photo *entity.Photo) []string {
	if photo == nil {
		return nil
	}

	var labels []classify.Label

	for _, label := range photo.ClassifyLabels() {
		switch entity.Src(label.Source) {
		case entity.SrcAuto, entity.SrcImage, entity.SrcONNX, entity.SrcPigo, entity.SrcOllama, entity.SrcOpenAI:
			continue
		}

		if strings.TrimSpace(label.Name) != "" {
			labels = append(labels, label)
		}
	}

	// Keep the most confident labels first, then cap the list.
	sort.SliceStable(labels, func(i, j int) bool {
		if labels[i].Confidence() == labels[j].Confidence() {
			return labels[i].Name < labels[j].Name
		}

		return labels[i].Confidence() > labels[j].Confidence()
	})

	if len(labels) > maxVisionLabels {
		labels = labels[:maxVisionLabels]
	}

	names := make([]string, 0, len(labels))

	for _, label := range labels {
		names = append(names, strings.TrimSpace(label.Name))
	}

	return names
}

// albumNamesForPhoto returns the titles of the visible albums the photo belongs to,
// capped so a many-to-many album list cannot bloat the prompt.
func albumNamesForPhoto(photo *entity.Photo) []string {
	if photo == nil || photo.PhotoUID == "" {
		return nil
	}

	var albums []entity.Album

	if err := entity.Db().
		Joins("JOIN photos_albums ON photos_albums.album_uid = albums.album_uid AND photos_albums.photo_uid = ?", photo.PhotoUID).
		Where("photos_albums.hidden = ?", false).
		Order("albums.album_title").
		Limit(3).
		Find(&albums).Error; err != nil {
		log.Debugf("vision: failed to load albums for %s (%s)", clean.Log(photo.String()), clean.Error(err))

		return nil
	}

	names := make([]string, 0, len(albums))

	for _, album := range albums {
		if title := strings.TrimSpace(album.AlbumTitle); title != "" {
			names = append(names, title)
		}
	}

	return names
}

// appendMetadata joins extra photo-level context onto the file-level metadata string.
func appendMetadata(metadata, extra string) string {
	if extra = strings.TrimSpace(extra); extra == "" {
		return metadata
	}

	if metadata != "" {
		return metadata + "; " + extra
	}

	return extra
}

// DetectNSFW returns an explicit safe, unsafe, or unavailable decision for the media file.
func (m *MediaFile) DetectNSFW() nsfw.Result {
	filename, err := m.Thumbnail(Config().ThumbCachePath(), thumb.Fit720)

	if err != nil {
		event.SystemError([]string{"vision", "%s in %s (detect nsfw)"}, clean.Error(err), clean.Log(m.RootRelName()))
		return nsfw.Unavailable(clean.Error(err))
	}

	results, modelErr := vision.DetectNSFW([]string{filename}, media.SrcLocal)

	switch {
	case modelErr != nil:
		event.SystemError([]string{"vision", "%s in %s (detect nsfw)"}, clean.Error(modelErr), clean.Log(m.RootRelName()))
		return nsfw.Unavailable(clean.Error(modelErr))
	case len(results) < 1:
		event.SystemError([]string{"vision", "nsfw model returned no result for %s"}, clean.Log(m.RootRelName()))
		return nsfw.Unavailable("no result")
	}

	if results[0].IsUnsafe() {
		log.Warnf("vision: detected offensive content in %s", clean.Log(m.RootRelName()))
	}

	return results[0]
}
