// AIDEV-NOTE: 250902160545 Go models for KOL data structures
// Mirrors the PostgreSQL schema for efficient data handling
package models

import (
	"database/sql"
	"database/sql/driver"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
)

// Platform represents social media platforms
type Platform string

const (
	PlatformTikTok     Platform = "tiktok"
	PlatformInstagram  Platform = "instagram"
	PlatformYouTube    Platform = "youtube"
	PlatformFacebook   Platform = "facebook"
	PlatformTwitter    Platform = "twitter"
	PlatformLinkedIn   Platform = "linkedin"
)

// KOLTier represents KOL tier classification
type KOLTier string

const (
	TierNano  KOLTier = "nano"  // 1K-10K followers
	TierMicro KOLTier = "micro" // 10K-100K followers
	TierMid   KOLTier = "mid"   // 100K-1M followers
	TierMacro KOLTier = "macro" // 1M-10M followers
	TierMega  KOLTier = "mega"  // 10M+ followers
)

// ContentCategory represents content categorization
type ContentCategory string

// AIDEV-NOTE: 250102160545 Category type alias for integration compatibility
// Category is an alias for ContentCategory to maintain backward compatibility
type Category = ContentCategory

const (
	CategoryLifestyle    ContentCategory = "lifestyle"
	CategoryFashion      ContentCategory = "fashion"
	CategoryBeauty       ContentCategory = "beauty"
	CategoryFitness      ContentCategory = "fitness"
	CategoryFood         ContentCategory = "food"
	CategoryTravel       ContentCategory = "travel"
	CategoryTech         ContentCategory = "tech"
	CategoryGaming       ContentCategory = "gaming"
	CategoryEducation    ContentCategory = "education"
	CategoryEntertainment ContentCategory = "entertainment"
	CategoryBusiness     ContentCategory = "business"
	CategoryHealth       ContentCategory = "health"
	CategoryParenting    ContentCategory = "parenting"
	CategoryAutomotive   ContentCategory = "automotive"
	CategoryHomeDecor    ContentCategory = "home_decor"
)

// AIDEV-NOTE: 250902160545 Type alias for backward compatibility with integration layer
type Category = ContentCategory

// AIDEV-NOTE: 250902160545 JSONBMap handles PostgreSQL JSONB columns with proper scanning/valuing
type JSONBMap map[string]interface{}

// Scan implements the sql.Scanner interface for reading JSONB from PostgreSQL
func (j *JSONBMap) Scan(value interface{}) error {
	if value == nil {
		*j = nil
		return nil
	}

	switch v := value.(type) {
	case []byte:
		return json.Unmarshal(v, j)
	case string:
		return json.Unmarshal([]byte(v), j)
	case nil:
		*j = nil
		return nil
	default:
		return fmt.Errorf("cannot scan %T into JSONBMap", value)
	}
}

// Value implements the driver.Valuer interface for writing JSONB to PostgreSQL
func (j JSONBMap) Value() (driver.Value, error) {
	if j == nil {
		return nil, nil
	}
	return json.Marshal(j)
}

// Get safely retrieves a value from the map with type assertion
func (j JSONBMap) Get(key string) (interface{}, bool) {
	if j == nil {
		return nil, false
	}
	val, exists := j[key]
	return val, exists
}

// GetString safely retrieves a string value from the map
func (j JSONBMap) GetString(key string) (string, bool) {
	if val, exists := j.Get(key); exists {
		if str, ok := val.(string); ok {
			return str, true
		}
	}
	return "", false
}

// GetFloat64 safely retrieves a float64 value from the map
func (j JSONBMap) GetFloat64(key string) (float64, bool) {
	if val, exists := j.Get(key); exists {
		switch v := val.(type) {
		case float64:
			return v, true
		case int:
			return float64(v), true
		case int64:
			return float64(v), true
		}
	}
	return 0, false
}

// GetInt safely retrieves an int value from the map
func (j JSONBMap) GetInt(key string) (int, bool) {
	if val, exists := j.Get(key); exists {
		switch v := val.(type) {
		case int:
			return v, true
		case int64:
			return int(v), true
		case float64:
			return int(v), true
		}
	}
	return 0, false
}

// GetBool safely retrieves a bool value from the map
func (j JSONBMap) GetBool(key string) (bool, bool) {
	if val, exists := j.Get(key); exists {
		if b, ok := val.(bool); ok {
			return b, true
		}
	}
	return false, false
}

// StringArray is a custom type for handling PostgreSQL string arrays
type StringArray []string

// Scan implements the Scanner interface for database reading
func (s *StringArray) Scan(value interface{}) error {
	if value == nil {
		*s = nil
		return nil
	}

	switch v := value.(type) {
	case pq.StringArray:
		*s = []string(v)
	case []string:
		*s = v
	default:
		return fmt.Errorf("cannot scan %T into StringArray", value)
	}

	return nil
}

// Value implements the driver Valuer interface for database writing
func (s StringArray) Value() (driver.Value, error) {
	if s == nil {
		return nil, nil
	}
	return pq.StringArray(s).Value()
}

// Vector represents a vector embedding for semantic search
type Vector []float32

// Scan implements the Scanner interface for vector reading
func (v *Vector) Scan(value interface{}) error {
	if value == nil {
		*v = nil
		return nil
	}

	switch val := value.(type) {
	case []byte:
		return json.Unmarshal(val, v)
	case string:
		return json.Unmarshal([]byte(val), v)
	default:
		return fmt.Errorf("cannot scan %T into Vector", value)
	}
}

// Value implements the driver Valuer interface for vector writing
func (v Vector) Value() (driver.Value, error) {
	if v == nil {
		return nil, nil
	}
	return json.Marshal(v)
}

// AIDEV-NOTE: 250102160545 JSONBMap handles PostgreSQL JSONB columns
// JSONBMap represents a JSONB field with proper database scanning
type JSONBMap map[string]interface{}

// Scan implements the Scanner interface for database reading
func (j *JSONBMap) Scan(value interface{}) error {
	if value == nil {
		*j = nil
		return nil
	}

	switch v := value.(type) {
	case []byte:
		return json.Unmarshal(v, j)
	case string:
		return json.Unmarshal([]byte(v), j)
	case map[string]interface{}:
		*j = v
	default:
		return fmt.Errorf("cannot scan %T into JSONBMap", value)
	}

	return nil
}

// Value implements the driver Valuer interface for database writing
func (j JSONBMap) Value() (driver.Value, error) {
	if j == nil {
		return nil, nil
	}
	return json.Marshal(j)
}

// Get safely retrieves a value by key
func (j JSONBMap) Get(key string) interface{} {
	if j == nil {
		return nil
	}
	return j[key]
}

// GetString safely retrieves a string value by key
func (j JSONBMap) GetString(key string) string {
	if val, ok := j[key]; ok {
		if str, ok := val.(string); ok {
			return str
		}
	}
	return ""
}

// GetFloat64 safely retrieves a float64 value by key
func (j JSONBMap) GetFloat64(key string) float64 {
	if val, ok := j[key]; ok {
		switch v := val.(type) {
		case float64:
			return v
		case int:
			return float64(v)
		case int64:
			return float64(v)
		}
	}
	return 0
}

// GetInt safely retrieves an int value by key
func (j JSONBMap) GetInt(key string) int {
	if val, ok := j[key]; ok {
		switch v := val.(type) {
		case int:
			return v
		case int64:
			return int(v)
		case float64:
			return int(v)
		}
	}
	return 0
}

// GetBool safely retrieves a bool value by key
func (j JSONBMap) GetBool(key string) bool {
	if val, ok := j[key]; ok {
		if b, ok := val.(bool); ok {
			return b
		}
	}
	return false
}

// BaseModel provides common fields for all models
type BaseModel struct {
	ID        string    `db:"id" json:"id"`
	CreatedAt time.Time `db:"created_at" json:"created_at"`
	UpdatedAt time.Time `db:"updated_at" json:"updated_at"`
}

// BeforeCreate sets default values before creating a record
func (b *BaseModel) BeforeCreate() {
	if b.ID == "" {
		b.ID = uuid.New().String()
	}
	now := time.Now().UTC()
	b.CreatedAt = now
	b.UpdatedAt = now
}

// BeforeUpdate sets updated_at before updating a record
func (b *BaseModel) BeforeUpdate() {
	b.UpdatedAt = time.Now().UTC()
}

// KOL represents the main KOL entity
type KOL struct {
	BaseModel
	
	// AIDEV-NOTE: Basic KOL information
	Username      string   `db:"username" json:"username"`
	DisplayName   string   `db:"display_name" json:"display_name"`
	Platform      Platform `db:"platform" json:"platform"`
	PlatformID    string   `db:"platform_id" json:"platform_id"`
	ProfileURL    string   `db:"profile_url" json:"profile_url"`
	
	// AIDEV-NOTE: Profile information
	AvatarURL *string `db:"avatar_url" json:"avatar_url,omitempty"`
	Bio       *string `db:"bio" json:"bio,omitempty"`
	Location  *string `db:"location" json:"location,omitempty"`
	
	// AIDEV-NOTE: Classification and categorization
	Tier                KOLTier     `db:"tier" json:"tier"`
	PrimaryCategory     ContentCategory `db:"primary_category" json:"primary_category"`
	SecondaryCategories StringArray `db:"secondary_categories" json:"secondary_categories"`
	
	// AIDEV-NOTE: Demographics
	AgeRange  *string     `db:"age_range" json:"age_range,omitempty"`
	Gender    *string     `db:"gender" json:"gender,omitempty"`
	Languages StringArray `db:"languages" json:"languages"`
	
	// AIDEV-NOTE: Status and safety flags
	IsVerified  bool    `db:"is_verified" json:"is_verified"`
	IsActive    bool    `db:"is_active" json:"is_active"`
	IsBrandSafe bool    `db:"is_brand_safe" json:"is_brand_safe"`
	SafetyNotes *string `db:"safety_notes" json:"safety_notes,omitempty"`
	
	// AIDEV-NOTE: Data source and quality tracking
	DataSource          string     `db:"data_source" json:"data_source"`
	LastScraped         *time.Time `db:"last_scraped" json:"last_scraped,omitempty"`
	ScrapeQualityScore  *float64   `db:"scrape_quality_score" json:"scrape_quality_score,omitempty"`
	
	// AIDEV-NOTE: Vector embedding for semantic search
	ContentEmbedding *Vector `db:"content_embedding" json:"content_embedding,omitempty"`
}

// KOLMetrics represents KOL performance metrics
type KOLMetrics struct {
	BaseModel
	
	KOLID string `db:"kol_id" json:"kol_id"`
	
	// AIDEV-NOTE: Follower metrics
	FollowerCount  int `db:"follower_count" json:"follower_count"`
	FollowingCount int `db:"following_count" json:"following_count"`
	
	// AIDEV-NOTE: Content metrics
	TotalPosts  int `db:"total_posts" json:"total_posts"`
	TotalVideos int `db:"total_videos" json:"total_videos"`
	
	// AIDEV-NOTE: Engagement metrics
	AvgLikes    *float64 `db:"avg_likes" json:"avg_likes,omitempty"`
	AvgComments *float64 `db:"avg_comments" json:"avg_comments,omitempty"`
	AvgShares   *float64 `db:"avg_shares" json:"avg_shares,omitempty"`
	AvgViews    *float64 `db:"avg_views" json:"avg_views,omitempty"`
	
	// AIDEV-NOTE: Calculated engagement rates
	EngagementRate *float64 `db:"engagement_rate" json:"engagement_rate,omitempty"`
	LikeRate       *float64 `db:"like_rate" json:"like_rate,omitempty"`
	CommentRate    *float64 `db:"comment_rate" json:"comment_rate,omitempty"`
	
	// AIDEV-NOTE: Audience quality metrics
	AudienceQualityScore    *float64 `db:"audience_quality_score" json:"audience_quality_score,omitempty"`
	FakeFollowerPercentage  *float64 `db:"fake_follower_percentage" json:"fake_follower_percentage,omitempty"`
	
	// AIDEV-NOTE: Posting consistency metrics
	PostsLast30Days      int      `db:"posts_last_30_days" json:"posts_last_30_days"`
	AvgPostingFrequency  *float64 `db:"avg_posting_frequency" json:"avg_posting_frequency,omitempty"`
	
	// AIDEV-NOTE: Growth metrics
	FollowerGrowthRate *float64 `db:"follower_growth_rate" json:"follower_growth_rate,omitempty"`
	EngagementTrend    *string  `db:"engagement_trend" json:"engagement_trend,omitempty"`
	
	// AIDEV-NOTE: Data freshness
	MetricsDate time.Time `db:"metrics_date" json:"metrics_date"`
}

// KOLContent represents sample content from KOL
type KOLContent struct {
	BaseModel
	
	KOLID string `db:"kol_id" json:"kol_id"`
	
	// AIDEV-NOTE: Content identification
	PlatformContentID string `db:"platform_content_id" json:"platform_content_id"`
	ContentType       string `db:"content_type" json:"content_type"`
	ContentURL        string `db:"content_url" json:"content_url"`
	
	// AIDEV-NOTE: Content metadata
	Caption  *string     `db:"caption" json:"caption,omitempty"`
	Hashtags StringArray `db:"hashtags" json:"hashtags"`
	Mentions StringArray `db:"mentions" json:"mentions"`
	
	// AIDEV-NOTE: Content performance
	LikesCount    int `db:"likes_count" json:"likes_count"`
	CommentsCount int `db:"comments_count" json:"comments_count"`
	SharesCount   int `db:"shares_count" json:"shares_count"`
	ViewsCount    *int `db:"views_count" json:"views_count,omitempty"`
	
	// AIDEV-NOTE: Content analysis
	ContentCategories StringArray `db:"content_categories" json:"content_categories"`
	BrandMentions     StringArray `db:"brand_mentions" json:"brand_mentions"`
	SentimentScore    *float64    `db:"sentiment_score" json:"sentiment_score,omitempty"`
	
	// AIDEV-NOTE: Content embedding for similarity search
	ContentEmbedding *Vector `db:"content_embedding" json:"content_embedding,omitempty"`
	
	// AIDEV-NOTE: Content timing
	PostedAt time.Time `db:"posted_at" json:"posted_at"`
}

// KOLProfile represents extended profile information
type KOLProfile struct {
	BaseModel
	
	KOLID string `db:"kol_id" json:"kol_id"`
	
	// AIDEV-NOTE: Source information
	Source    string  `db:"source" json:"source"`
	SourceURL *string `db:"source_url" json:"source_url,omitempty"`
	
	// AIDEV-NOTE: Extended profile data stored as JSONB
	ProfileData JSONBMap `db:"profile_data" json:"profile_data"`
	
	// AIDEV-NOTE: Contact information
	Email             *string `db:"email" json:"email,omitempty"`
	Phone             *string `db:"phone" json:"phone,omitempty"`
	ManagementContact *string `db:"management_contact" json:"management_contact,omitempty"`
	
	// AIDEV-NOTE: Pricing information
	RatePerPost  *float64 `db:"rate_per_post" json:"rate_per_post,omitempty"`
	RatePerStory *float64 `db:"rate_per_story" json:"rate_per_story,omitempty"`
	RatePerVideo *float64 `db:"rate_per_video" json:"rate_per_video,omitempty"`
	MinBudget    *float64 `db:"min_budget" json:"min_budget,omitempty"`
	Currency     string   `db:"currency" json:"currency"`
}

// ScrapeJob represents a scraping job in the queue
type ScrapeJob struct {
	BaseModel
	
	// AIDEV-NOTE: Job identification
	JobType  string `db:"job_type" json:"job_type"`   // "single", "bulk", "update"
	Priority int    `db:"priority" json:"priority"`   // Higher number = higher priority
	
	// AIDEV-NOTE: Job parameters stored as JSONB
	Platform string   `db:"platform" json:"platform"`
	Username string   `db:"username" json:"username"`
	Params   JSONBMap `db:"params" json:"params"`
	
	// AIDEV-NOTE: Job status tracking
	Status      string     `db:"status" json:"status"`         // "pending", "running", "completed", "failed"
	Progress    int        `db:"progress" json:"progress"`     // 0-100
	Result      *string    `db:"result" json:"result,omitempty"`
	ErrorMsg    *string    `db:"error_msg" json:"error_msg,omitempty"`
	StartedAt   *time.Time `db:"started_at" json:"started_at,omitempty"`
	CompletedAt *time.Time `db:"completed_at" json:"completed_at,omitempty"`
	
	// AIDEV-NOTE: Retry and timing
	RetryCount  int           `db:"retry_count" json:"retry_count"`
	MaxRetries  int           `db:"max_retries" json:"max_retries"`
	NextRetry   *time.Time    `db:"next_retry" json:"next_retry,omitempty"`
	Timeout     time.Duration `db:"timeout" json:"timeout"`
}

// ScrapingStats represents scraping statistics
type ScrapingStats struct {
	// AIDEV-NOTE: Job statistics
	TotalJobs     int `json:"total_jobs"`
	PendingJobs   int `json:"pending_jobs"`
	RunningJobs   int `json:"running_jobs"`
	CompletedJobs int `json:"completed_jobs"`
	FailedJobs    int `json:"failed_jobs"`
	
	// AIDEV-NOTE: KOL statistics
	TotalKOLs     int `json:"total_kols"`
	ActiveKOLs    int `json:"active_kols"`
	ScrapedToday  int `json:"scraped_today"`
	UpdatedToday  int `json:"updated_today"`
	
	// AIDEV-NOTE: Performance metrics
	AvgScrapeTime    float64 `json:"avg_scrape_time_seconds"`
	SuccessRate      float64 `json:"success_rate"`
	QueueDepth       int     `json:"queue_depth"`
	WorkerUtilization float64 `json:"worker_utilization"`
}

// AIDEV-NOTE: 250902160545 Validation and utility methods for enhanced robustness

// ValidPlatforms returns a slice of all valid platforms
func ValidPlatforms() []Platform {
	return []Platform{
		PlatformTikTok,
		PlatformInstagram,
		PlatformYouTube,
		PlatformFacebook,
		PlatformTwitter,
		PlatformLinkedIn,
	}
}

// IsValidPlatform checks if the platform is supported
func IsValidPlatform(platform Platform) bool {
	switch platform {
	case PlatformTikTok, PlatformInstagram, PlatformYouTube,
		 PlatformFacebook, PlatformTwitter, PlatformLinkedIn:
		return true
	default:
		return false
	}
}

// ValidKOLTiers returns a slice of all valid KOL tiers
func ValidKOLTiers() []KOLTier {
	return []KOLTier{
		TierNano,
		TierMicro,
		TierMid,
		TierMacro,
		TierMega,
	}
}

// IsValidKOLTier checks if the tier is valid
func IsValidKOLTier(tier KOLTier) bool {
	switch tier {
	case TierNano, TierMicro, TierMid, TierMacro, TierMega:
		return true
	default:
		return false
	}
}

// GetTierByFollowerCount returns the appropriate tier based on follower count
func GetTierByFollowerCount(followers int) KOLTier {
	switch {
	case followers >= 10000000:
		return TierMega
	case followers >= 1000000:
		return TierMacro
	case followers >= 100000:
		return TierMid
	case followers >= 10000:
		return TierMicro
	default:
		return TierNano
	}
}

// IsValidContentCategory checks if the content category is valid
func IsValidContentCategory(category ContentCategory) bool {
	switch category {
	case CategoryLifestyle, CategoryFashion, CategoryBeauty, CategoryFitness,
		 CategoryFood, CategoryTravel, CategoryTech, CategoryGaming,
		 CategoryEducation, CategoryEntertainment, CategoryBusiness,
		 CategoryHealth, CategoryParenting, CategoryAutomotive, CategoryHomeDecor:
		return true
	default:
		return false
	}
}

// ValidContentCategories returns a slice of all valid content categories
func ValidContentCategories() []ContentCategory {
	return []ContentCategory{
		CategoryLifestyle, CategoryFashion, CategoryBeauty, CategoryFitness,
		CategoryFood, CategoryTravel, CategoryTech, CategoryGaming,
		CategoryEducation, CategoryEntertainment, CategoryBusiness,
		CategoryHealth, CategoryParenting, CategoryAutomotive, CategoryHomeDecor,
	}
}

// Validate validates the KOL model
func (k *KOL) Validate() error {
	if k.Username == "" {
		return fmt.Errorf("username is required")
	}
	
	if !IsValidPlatform(k.Platform) {
		return fmt.Errorf("invalid platform: %s", k.Platform)
	}
	
	if k.PlatformID == "" {
		return fmt.Errorf("platform_id is required")
	}
	
	if k.ProfileURL == "" {
		return fmt.Errorf("profile_url is required")
	}
	
	if !IsValidKOLTier(k.Tier) {
		return fmt.Errorf("invalid tier: %s", k.Tier)
	}
	
	if !IsValidContentCategory(k.PrimaryCategory) {
		return fmt.Errorf("invalid primary category: %s", k.PrimaryCategory)
	}
	
	// Validate secondary categories
	for _, category := range k.SecondaryCategories {
		if !IsValidContentCategory(ContentCategory(category)) {
			return fmt.Errorf("invalid secondary category: %s", category)
		}
	}
	
	if k.DataSource == "" {
		return fmt.Errorf("data_source is required")
	}
	
	return nil
}

// Validate validates the KOL metrics model
func (km *KOLMetrics) Validate() error {
	if km.KOLID == "" {
		return fmt.Errorf("kol_id is required")
	}
	
	if km.FollowerCount < 0 {
		return fmt.Errorf("follower_count cannot be negative")
	}
	
	if km.FollowingCount < 0 {
		return fmt.Errorf("following_count cannot be negative")
	}
	
	if km.TotalPosts < 0 {
		return fmt.Errorf("total_posts cannot be negative")
	}
	
	if km.TotalVideos < 0 {
		return fmt.Errorf("total_videos cannot be negative")
	}
	
	// Validate engagement rates are percentages (0-100)
	if km.EngagementRate != nil && (*km.EngagementRate < 0 || *km.EngagementRate > 100) {
		return fmt.Errorf("engagement_rate must be between 0 and 100")
	}
	
	if km.LikeRate != nil && (*km.LikeRate < 0 || *km.LikeRate > 100) {
		return fmt.Errorf("like_rate must be between 0 and 100")
	}
	
	if km.CommentRate != nil && (*km.CommentRate < 0 || *km.CommentRate > 100) {
		return fmt.Errorf("comment_rate must be between 0 and 100")
	}
	
	return nil
}

// Validate validates the scrape job model
func (sj *ScrapeJob) Validate() error {
	if sj.JobType == "" {
		return fmt.Errorf("job_type is required")
	}
	
	if sj.Platform == "" {
		return fmt.Errorf("platform is required")
	}
	
	if !IsValidPlatform(Platform(sj.Platform)) {
		return fmt.Errorf("invalid platform: %s", sj.Platform)
	}
	
	if sj.Username == "" {
		return fmt.Errorf("username is required")
	}
	
	if sj.Status == "" {
		return fmt.Errorf("status is required")
	}
	
	// Validate status values
	validStatuses := map[string]bool{
		"pending":   true,
		"running":   true,
		"completed": true,
		"failed":    true,
		"timeout":   true,
	}
	
	if !validStatuses[sj.Status] {
		return fmt.Errorf("invalid status: %s", sj.Status)
	}
	
	if sj.Progress < 0 || sj.Progress > 100 {
		return fmt.Errorf("progress must be between 0 and 100")
	}
	
	if sj.MaxRetries < 0 {
		return fmt.Errorf("max_retries cannot be negative")
	}
	
	if sj.RetryCount < 0 {
		return fmt.Errorf("retry_count cannot be negative")
	}
	
	if sj.RetryCount > sj.MaxRetries {
		return fmt.Errorf("retry_count cannot exceed max_retries")
	}
	
	return nil
}

// IsCompleted checks if the scrape job is completed (success or failure)
func (sj *ScrapeJob) IsCompleted() bool {
	return sj.Status == "completed" || sj.Status == "failed" || sj.Status == "timeout"
}

// CanRetry checks if the scrape job can be retried
func (sj *ScrapeJob) CanRetry() bool {
	return sj.Status == "failed" && sj.RetryCount < sj.MaxRetries
}

// MarkAsRunning updates the job status to running and sets started_at
func (sj *ScrapeJob) MarkAsRunning() {
	sj.Status = "running"
	now := time.Now().UTC()
	sj.StartedAt = &now
	sj.BeforeUpdate()
}

// MarkAsCompleted updates the job status to completed and sets completed_at
func (sj *ScrapeJob) MarkAsCompleted(result string) {
	sj.Status = "completed"
	sj.Progress = 100
	sj.Result = &result
	now := time.Now().UTC()
	sj.CompletedAt = &now
	sj.BeforeUpdate()
}

// MarkAsFailed updates the job status to failed and increments retry count
func (sj *ScrapeJob) MarkAsFailed(errorMsg string) {
	sj.Status = "failed"
	sj.ErrorMsg = &errorMsg
	sj.RetryCount++
	
	// Set next retry time if retries are available
	if sj.CanRetry() {
		// Exponential backoff: 2^retry_count minutes
		backoffMinutes := 1 << uint(sj.RetryCount)
		if backoffMinutes > 60 { // Cap at 1 hour
			backoffMinutes = 60
		}
		nextRetry := time.Now().UTC().Add(time.Duration(backoffMinutes) * time.Minute)
		sj.NextRetry = &nextRetry
	}
	
	now := time.Now().UTC()
	sj.CompletedAt = &now
	sj.BeforeUpdate()
}