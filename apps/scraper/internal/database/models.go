// AIDEV-NOTE: Database models that map Go structs to PostgreSQL tables
// Re-exports the models from internal/models for database operations
package database

import "kol-scraper/internal/models"

// Re-export model types for convenience
type (
	KOL         = models.KOL
	KOLMetrics  = models.KOLMetrics
	KOLContent  = models.KOLContent
	KOLProfile  = models.KOLProfile
	ScrapeJob   = models.ScrapeJob
	StringArray = models.StringArray
	Vector      = models.Vector
	Platform    = models.Platform
	KOLTier     = models.KOLTier
	ContentCategory = models.ContentCategory
)

// Platform constants
const (
	PlatformTikTok     = models.PlatformTikTok
	PlatformInstagram  = models.PlatformInstagram
	PlatformYouTube    = models.PlatformYouTube
	PlatformFacebook   = models.PlatformFacebook
	PlatformTwitter    = models.PlatformTwitter
	PlatformLinkedIn   = models.PlatformLinkedIn
)

// Tier constants
const (
	TierNano  = models.TierNano
	TierMicro = models.TierMicro
	TierMid   = models.TierMid
	TierMacro = models.TierMacro
	TierMega  = models.TierMega
)

// Category constants
const (
	CategoryLifestyle    = models.CategoryLifestyle
	CategoryFashion      = models.CategoryFashion
	CategoryBeauty       = models.CategoryBeauty
	CategoryFitness      = models.CategoryFitness
	CategoryFood         = models.CategoryFood
	CategoryTravel       = models.CategoryTravel
	CategoryTech         = models.CategoryTech
	CategoryGaming       = models.CategoryGaming
	CategoryEducation    = models.CategoryEducation
	CategoryEntertainment = models.CategoryEntertainment
	CategoryBusiness     = models.CategoryBusiness
	CategoryHealth       = models.CategoryHealth
	CategoryParenting    = models.CategoryParenting
	CategoryAutomotive   = models.CategoryAutomotive
	CategoryHomeDecor    = models.CategoryHomeDecor
)