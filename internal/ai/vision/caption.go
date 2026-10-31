package vision

import (
	"context"
	"errors"

	"github.com/photoprism/photoprism/pkg/media"
)

var captionFunc = captionInternal

// SetCaptionFunc overrides the caption generator. Intended for tests.
func SetCaptionFunc(fn func(Files, media.Src, string) (*CaptionResult, *Model, error)) {
	if fn == nil {
		captionFunc = captionInternal
		return
	}

	captionFunc = fn
}

// GenerateCaption returns generated captions for the specified images.
// metadata may hold data-only file context (date, place, camera) appended to the model prompt.
func GenerateCaption(images Files, mediaSrc media.Src, metadata string) (*CaptionResult, *Model, error) {
	return captionFunc(images, mediaSrc, metadata)
}

func captionInternal(images Files, mediaSrc media.Src, metadata string) (result *CaptionResult, model *Model, err error) {
	// Return if there is no configuration or no image classification models are configured.
	if Config == nil {
		return result, model, errors.New("vision service is not configured")
	} else if model = Config.Model(ModelTypeCaption); model != nil {
		// Refuse a model whose own service URI does not resolve.
		if err = model.unresolvedUriErr(); err != nil {
			return result, model, err
		}

		// Use remote service API if a server endpoint has been configured.
		if uri, method := model.Endpoint(); uri != "" && method != "" {
			var apiRequest *ApiRequest
			var apiResponse *ApiResponse

			if engine, ok := EngineFor(model.EndpointRequestFormat()); ok && engine.Builder != nil {
				if apiRequest, err = engine.Builder.Build(context.Background(), model, images, mediaSrc); err != nil {
					return result, model, err
				}
			} else if apiRequest, err = NewApiRequest(model.EndpointRequestFormat(), images, model.EndpointFileScheme(), mediaSrc); err != nil {
				return result, model, err
			}

			if apiRequest.Model == "" {
				apiRequest.Model, _, apiRequest.Version = model.GetModel()
			}

			model.ApplyService(apiRequest)

			apiRequest.System = model.GetSystemPrompt()
			apiRequest.Prompt = PromptWithMetadata(model.GetPrompt(), metadata)

			if apiRequest.Options == nil {
				apiRequest.Options = model.GetOptions()
			}

			apiRequest.WriteLog()

			perform := func(a *ApiRequest) (*ApiResponse, error) {
				return PerformApiRequest(a, uri, method, model.EndpointKey())
			}
			apiResponse, err = performVisionRequest(apiRequest, "caption", perform, func(r *ApiResponse) bool { return r.Result.Caption != nil })
			if err != nil {
				return result, model, err
			}
			if apiResponse.Result.Caption == nil {
				return result, model, errors.New("invalid caption model response")
			}

			// Set image as the default caption source.
			if apiResponse.Result.Caption.Source == "" {
				apiResponse.Result.Caption.Source = model.GetSource()
			}

			result = apiResponse.Result.Caption
			result.Model = apiRequest.Model
		} else {
			return result, model, errors.New("invalid caption model configuration")
		}
	} else {
		return result, model, errors.New("missing caption model")
	}

	return result, model, nil
}
