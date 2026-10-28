package api

import (
	"errors"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/photoprism/photoprism/internal/ai/vision"
	"github.com/photoprism/photoprism/internal/auth/acl"
	"github.com/photoprism/photoprism/internal/config"
	"github.com/photoprism/photoprism/internal/entity"
	"github.com/photoprism/photoprism/internal/mutex"
	"github.com/photoprism/photoprism/internal/photoprism/get"
	"github.com/photoprism/photoprism/internal/workers"
	"github.com/photoprism/photoprism/pkg/clean"
	"github.com/photoprism/photoprism/pkg/i18n"
)

// batchPhotosCaptionRequest carries the photos to caption and whether existing captions are replaced.
type batchPhotosCaptionRequest struct {
	Photos []string `json:"photos"`
	Force  bool     `json:"force"`
}

// BatchPhotosCaption queues AI caption generation for a set of photos.
//
//	@Summary	queues AI caption generation for a set of photos
//	@Id		BatchPhotosCaption
//	@Tags		Photos
//	@Accept		json
//	@Produce	json
//	@Success	202	{object}	map[string]string	"confirmation message"
//	@Failure	400,401,403,412,413,429,500	{object}	i18n.Response
//	@Param		Request	body		batchPhotosCaptionRequest	true	"photo UIDs and force flag"
//	@Router		/api/v1/batch/photos/caption [post]
func BatchPhotosCaption(router *gin.RouterGroup) {
	router.POST("/batch/photos/caption", func(c *gin.Context) {
		// Require access to all photos.
		s := Auth(c, acl.ResourcePhotos, acl.AccessAll)

		if s.Abort(c) {
			return
		}

		// Require update permissions for photos.
		if acl.Rules.Deny(acl.ResourcePhotos, s.GetUserRole(), acl.ActionUpdate) {
			AbortForbidden(c)
			return
		}

		// Reject queued runs while another vision job is active.
		if mutex.VisionWorker.Running() {
			AbortBusy(c)
			return
		}

		var frm batchPhotosCaptionRequest

		// Assign and validate request form values.
		LimitRequestBodyBytes(c, MaxBatchPhotosEditBytes)

		if err := c.BindJSON(&frm); err != nil {
			if IsRequestBodyTooLarge(err) {
				AbortRequestTooLarge(c, i18n.ErrBadRequest)
				return
			}

			AbortBadRequest(c, err)
			return
		}

		if len(frm.Photos) == 0 {
			AbortBadRequest(c, errors.New("no photos provided"))
			return
		}

		conf := get.Config()

		// Refuse the run when no caption model is configured.
		if !conf.VisionModelShouldRun(vision.ModelTypeCaption, vision.RunManual) {
			Abort(c, http.StatusPreconditionFailed, i18n.ErrNoCaptionModel)
			return
		}

		// Generate captions in the background; results are announced per photo via entity events.
		go startPhotoCaptions(conf, frm)

		c.JSON(http.StatusAccepted, gin.H{"message": "caption generation started"})
	})
}

// startPhotoCaptions runs the vision worker for the requested photos and logs failures.
func startPhotoCaptions(conf *config.Config, frm batchPhotosCaptionRequest) {
	filter := "uid:" + strings.Join(frm.Photos, "|")

	err := workers.NewVision(conf).Start(filter, len(frm.Photos), []string{vision.ModelTypeCaption}, entity.SrcAuto, frm.Force, vision.RunManual)

	if err != nil {
		log.Errorf("vision: %s (queued caption run)", clean.ErrorFull(err))
	}
}
