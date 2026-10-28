package vision

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPromptWithMetadata(t *testing.T) {
	t.Run("EmptyMetadataReturnsPromptUnchanged", func(t *testing.T) {
		assert.Equal(t, "base prompt", PromptWithMetadata("base prompt", ""))
		assert.Equal(t, "base prompt", PromptWithMetadata("base prompt", "   \n\t "))
	})
	t.Run("EmptyPromptReturnsHeaderOnly", func(t *testing.T) {
		assert.Equal(t, MetadataPromptHeader+"\nTaken 2024-06-15", PromptWithMetadata("", "Taken 2024-06-15"))
	})
	t.Run("AppendsMetadataBlock", func(t *testing.T) {
		expected := "base prompt\n\n" + MetadataPromptHeader + "\nTaken 2024-06-15; Place Berlin, Germany"
		assert.Equal(t, expected, PromptWithMetadata("base prompt", "Taken 2024-06-15; Place Berlin, Germany"))
	})
	t.Run("TrimsMetadataWhitespace", func(t *testing.T) {
		expected := "base prompt\n\n" + MetadataPromptHeader + "\nCamera Fuji X-T5"
		assert.Equal(t, expected, PromptWithMetadata("base prompt", "  Camera Fuji X-T5\n"))
	})
}
