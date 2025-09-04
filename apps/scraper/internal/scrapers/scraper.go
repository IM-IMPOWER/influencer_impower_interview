// AIDEV-NOTE: Scraper interface and manager for multi-platform support
// Provides unified interface for different social media platform scrapers
package scrapers

import (
	"context"
	"fmt"

	"kol-scraper/internal/models"
	"kol-scraper/pkg/config"
	"kol-scraper/pkg/logger"
)

// Scraper defines the interface for platform-specific scrapers
type Scraper interface {
	// ScrapeProfile scrapes a user profile and returns structured data
	ScrapeProfile(ctx context.Context, username string) (*ScrapeResult, error)
	
	// ConvertToKOLModel converts scrape result to database models
	ConvertToKOLModel(result *ScrapeResult, username string) (*models.KOL, *models.KOLMetrics, []*models.KOLContent)
}

// Manager manages multiple platform scrapers
type Manager struct {
	scrapers map[models.Platform]Scraper
	logger   *logger.Logger
}

// NewManager creates a new scraper manager with configured scrapers
func NewManager(cfg *config.Config, log *logger.Logger) *Manager {
	manager := &Manager{
		scrapers: make(map[models.Platform]Scraper),
		logger:   log,
	}

	// AIDEV-NOTE: Initialize TikTok scraper if enabled
	if cfg.TikTokConfig.Enabled {
		tiktokScraper := NewTikTokScraper(&cfg.TikTokConfig, log, cfg.UserAgent)
		manager.scrapers[models.PlatformTikTok] = tiktokScraper
		log.Info("TikTok scraper initialized")
	}

	// AIDEV-NOTE: TODO - Add other platform scrapers here
	// manager.scrapers[models.PlatformInstagram] = NewInstagramScraper(cfg, log)
	// manager.scrapers[models.PlatformYouTube] = NewYouTubeScraper(cfg, log)

	return manager
}

// ScrapeProfile scrapes a profile using the appropriate platform scraper
func (m *Manager) ScrapeProfile(ctx context.Context, platform models.Platform, username string) (*ScrapeResult, error) {
	scraper, exists := m.scrapers[platform]
	if !exists {
		return nil, fmt.Errorf("scraper for platform %s not available", platform)
	}

	m.logger.ScrapeLog(string(platform), username, "scrape_start", logger.Fields{
		"scraper_type": "platform_specific",
	})

	result, err := scraper.ScrapeProfile(ctx, username)
	if err != nil {
		m.logger.ScrapeLog(string(platform), username, "scrape_error", logger.Fields{
			"error": err.Error(),
		})
		return nil, err
	}

	m.logger.ScrapeLog(string(platform), username, "scrape_completed", logger.Fields{
		"has_profile": result.Profile != nil,
		"videos_count": len(result.Videos),
	})

	return result, nil
}

// ConvertToModels converts scrape result to database models using the appropriate scraper
func (m *Manager) ConvertToModels(platform models.Platform, result *ScrapeResult, username string) (*models.KOL, *models.KOLMetrics, []*models.KOLContent, error) {
	scraper, exists := m.scrapers[platform]
	if !exists {
		return nil, nil, nil, fmt.Errorf("scraper for platform %s not available", platform)
	}

	kol, metrics, content := scraper.ConvertToKOLModel(result, username)
	if kol == nil {
		return nil, nil, nil, fmt.Errorf("failed to convert scrape result to models")
	}

	m.logger.ScrapeLog(string(platform), username, "models_converted", logger.Fields{
		"kol_id": kol.ID,
		"metrics_available": metrics != nil,
		"content_items": len(content),
	})

	return kol, metrics, content, nil
}

// GetAvailablePlatforms returns list of available platform scrapers
func (m *Manager) GetAvailablePlatforms() []models.Platform {
	platforms := make([]models.Platform, 0, len(m.scrapers))
	for platform := range m.scrapers {
		platforms = append(platforms, platform)
	}
	return platforms
}

// IsSupported checks if a platform is supported
func (m *Manager) IsSupported(platform models.Platform) bool {
	_, exists := m.scrapers[platform]
	return exists
}

// GetScraperInfo returns information about available scrapers
func (m *Manager) GetScraperInfo() map[string]interface{} {
	info := map[string]interface{}{
		"available_platforms": []string{},
		"total_scrapers": len(m.scrapers),
	}

	platforms := make([]string, 0, len(m.scrapers))
	for platform := range m.scrapers {
		platforms = append(platforms, string(platform))
	}
	info["available_platforms"] = platforms

	return info
}

// HealthCheck checks the health of all scrapers
func (m *Manager) HealthCheck(ctx context.Context) map[string]bool {
	health := make(map[string]bool)

	for platform := range m.scrapers {
		// AIDEV-NOTE: Simple health check - could be enhanced with actual scraper testing
		health[string(platform)] = true
	}

	return health
}

// ValidateUsername validates username format for a specific platform
func (m *Manager) ValidateUsername(platform models.Platform, username string) error {
	if username == "" {
		return fmt.Errorf("username cannot be empty")
	}

	// AIDEV-NOTE: Platform-specific username validation
	switch platform {
	case models.PlatformTikTok:
		return validateTikTokUsername(username)
	case models.PlatformInstagram:
		return validateInstagramUsername(username)
	case models.PlatformYouTube:
		return validateYouTubeUsername(username)
	default:
		return fmt.Errorf("platform %s not supported", platform)
	}
}

// AIDEV-NOTE: Platform-specific username validation functions

func validateTikTokUsername(username string) error {
	// TikTok usernames can contain letters, numbers, underscores, and periods
	// Must be 2-24 characters long
	if len(username) < 2 || len(username) > 24 {
		return fmt.Errorf("TikTok username must be 2-24 characters long")
	}

	for _, char := range username {
		if !((char >= 'a' && char <= 'z') || 
			 (char >= 'A' && char <= 'Z') || 
			 (char >= '0' && char <= '9') || 
			 char == '_' || char == '.') {
			return fmt.Errorf("TikTok username contains invalid characters")
		}
	}

	return nil
}

func validateInstagramUsername(username string) error {
	// Instagram usernames can contain letters, numbers, underscores, and periods
	// Must be 1-30 characters long
	if len(username) < 1 || len(username) > 30 {
		return fmt.Errorf("Instagram username must be 1-30 characters long")
	}

	for _, char := range username {
		if !((char >= 'a' && char <= 'z') || 
			 (char >= 'A' && char <= 'Z') || 
			 (char >= '0' && char <= '9') || 
			 char == '_' || char == '.') {
			return fmt.Errorf("Instagram username contains invalid characters")
		}
	}

	return nil
}

func validateYouTubeUsername(username string) error {
	// YouTube channel names/usernames have different rules
	// Can be channel ID, custom URL, or username
	if len(username) < 1 || len(username) > 100 {
		return fmt.Errorf("YouTube username must be 1-100 characters long")
	}

	return nil
}

// Error types for better error handling
type ScraperError struct {
	Platform string
	Username string
	Message  string
	Err      error
}

func (e *ScraperError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("scraper error for %s/%s: %s: %v", e.Platform, e.Username, e.Message, e.Err)
	}
	return fmt.Sprintf("scraper error for %s/%s: %s", e.Platform, e.Username, e.Message)
}

func (e *ScraperError) Unwrap() error {
	return e.Err
}

// NewScraperError creates a new scraper error
func NewScraperError(platform, username, message string, err error) *ScraperError {
	return &ScraperError{
		Platform: platform,
		Username: username,
		Message:  message,
		Err:      err,
	}
}