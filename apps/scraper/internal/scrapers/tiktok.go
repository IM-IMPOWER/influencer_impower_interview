// AIDEV-NOTE: TikTok scraper implementation with rate limiting and retry logic
// Handles profile data extraction while respecting platform limits
package scrapers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
	"github.com/chromedp/chromedp"
	"golang.org/x/time/rate"
	"kol-scraper/internal/models"
	"kol-scraper/pkg/config"
	"kol-scraper/pkg/logger"
)

// TikTokScraper handles TikTok data extraction
type TikTokScraper struct {
	config      *config.TikTokConfig
	rateLimiter *rate.Limiter
	httpClient  *http.Client
	logger      *logger.Logger
	userAgent   string
}

// TikTokProfile represents raw TikTok profile data
type TikTokProfile struct {
	Username        string `json:"username"`
	DisplayName     string `json:"displayName"`
	Bio             string `json:"bio"`
	FollowerCount   int    `json:"followerCount"`
	FollowingCount  int    `json:"followingCount"`
	LikesCount      int    `json:"likesCount"`
	VideoCount      int    `json:"videoCount"`
	IsVerified      bool   `json:"isVerified"`
	AvatarURL       string `json:"avatarURL"`
	ProfileURL      string `json:"profileURL"`
	Location        string `json:"location"`
	Website         string `json:"website"`
}

// TikTokVideo represents a TikTok video with metrics
type TikTokVideo struct {
	ID            string   `json:"id"`
	Description   string   `json:"description"`
	URL           string   `json:"url"`
	LikesCount    int      `json:"likesCount"`
	CommentsCount int      `json:"commentsCount"`
	SharesCount   int      `json:"sharesCount"`
	ViewsCount    int      `json:"viewsCount"`
	PostedAt      string   `json:"postedAt"`
	Hashtags      []string `json:"hashtags"`
	Mentions      []string `json:"mentions"`
}

// ScrapeResult contains the complete scraping result
type ScrapeResult struct {
	Profile *TikTokProfile `json:"profile"`
	Videos  []TikTokVideo  `json:"videos"`
	Error   error          `json:"error,omitempty"`
}

// NewTikTokScraper creates a new TikTok scraper instance
func NewTikTokScraper(cfg *config.TikTokConfig, log *logger.Logger, userAgent string) *TikTokScraper {
	// AIDEV-NOTE: Create rate limiter based on configuration
	rateLimiter := rate.NewLimiter(rate.Limit(cfg.RateLimitRPS), 1)

	// AIDEV-NOTE: Configure HTTP client with timeout and proxy support
	client := &http.Client{
		Timeout: cfg.Timeout,
	}

	if cfg.ProxyURL != "" {
		if proxyURL, err := url.Parse(cfg.ProxyURL); err == nil {
			client.Transport = &http.Transport{
				Proxy: http.ProxyURL(proxyURL),
			}
		}
	}

	return &TikTokScraper{
		config:      cfg,
		rateLimiter: rateLimiter,
		httpClient:  client,
		logger:      log,
		userAgent:   userAgent,
	}
}

// ScrapeProfile scrapes a TikTok profile and returns structured data
func (ts *TikTokScraper) ScrapeProfile(ctx context.Context, username string) (*ScrapeResult, error) {
	// AIDEV-NOTE: Wait for rate limiter before proceeding
	if err := ts.rateLimiter.Wait(ctx); err != nil {
		return nil, fmt.Errorf("rate limiter wait failed: %w", err)
	}

	ts.logger.ScrapeLog("tiktok", username, "scrape_start", logger.Fields{})

	// AIDEV-NOTE: Try multiple scraping methods with fallback
	result, err := ts.scrapeWithChrome(ctx, username)
	if err != nil {
		ts.logger.ScrapeLog("tiktok", username, "chrome_scrape_failed", logger.Fields{
			"error": err.Error(),
		})

		// AIDEV-NOTE: Fallback to HTTP-only scraping
		result, err = ts.scrapeWithHTTP(ctx, username)
		if err != nil {
			ts.logger.ScrapeLog("tiktok", username, "http_scrape_failed", logger.Fields{
				"error": err.Error(),
			})
			return nil, fmt.Errorf("all scraping methods failed: %w", err)
		}
	}

	ts.logger.ScrapeLog("tiktok", username, "scrape_completed", logger.Fields{
		"videos_found": len(result.Videos),
	})

	return result, nil
}

// scrapeWithChrome uses headless Chrome for dynamic content scraping
func (ts *TikTokScraper) scrapeWithChrome(ctx context.Context, username string) (*ScrapeResult, error) {
	// AIDEV-NOTE: Create Chrome context with timeout
	chromeCtx, cancel := chromedp.NewContext(ctx)
	defer cancel()

	profileURL := fmt.Sprintf("https://www.tiktok.com/@%s", username)
	
	var profileData string
	var videosData string

	// AIDEV-NOTE: Navigate to profile and extract data
	err := chromedp.Run(chromeCtx,
		chromedp.Navigate(profileURL),
		chromedp.WaitVisible(`[data-e2e="user-title"]`, chromedp.ByQuery),
		chromedp.Sleep(2*time.Second),
		chromedp.EvaluateAsDevTools(`
			JSON.stringify({
				profile: window.__INITIAL_STATE__ || window.__NEXT_DATA__ || {},
				videos: Array.from(document.querySelectorAll('[data-e2e="user-post-item"]')).slice(0, 10).map(item => {
					const link = item.querySelector('a');
					const desc = item.querySelector('[data-e2e="user-post-item-desc"]');
					return {
						url: link ? link.href : '',
						description: desc ? desc.textContent : '',
						element: item.outerHTML
					};
				})
			})
		`, &profileData),
	)

	if err != nil {
		return nil, fmt.Errorf("chrome scraping failed: %w", err)
	}

	return ts.parseScrapedData(profileData, username)
}

// scrapeWithHTTP uses HTTP-only scraping for basic profile information
func (ts *TikTokScraper) scrapeWithHTTP(ctx context.Context, username string) (*ScrapeResult, error) {
	profileURL := fmt.Sprintf("https://www.tiktok.com/@%s", username)

	// AIDEV-NOTE: Create HTTP request with proper headers
	req, err := http.NewRequestWithContext(ctx, "GET", profileURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("User-Agent", ts.userAgent)
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
	req.Header.Set("Accept-Language", "en-US,en;q=0.5")
	req.Header.Set("Accept-Encoding", "gzip, deflate")
	req.Header.Set("DNT", "1")
	req.Header.Set("Connection", "keep-alive")
	req.Header.Set("Upgrade-Insecure-Requests", "1")

	if ts.config.SessionCookie != "" {
		req.Header.Set("Cookie", ts.config.SessionCookie)
	}

	// AIDEV-NOTE: Execute request with retries
	var resp *http.Response
	for attempt := 0; attempt < ts.config.MaxRetries; attempt++ {
		resp, err = ts.httpClient.Do(req)
		if err != nil {
			if attempt == ts.config.MaxRetries-1 {
				return nil, fmt.Errorf("HTTP request failed after %d attempts: %w", ts.config.MaxRetries, err)
			}
			time.Sleep(time.Duration(attempt+1) * time.Second)
			continue
		}
		break
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP request failed with status %d", resp.StatusCode)
	}

	// AIDEV-NOTE: Parse HTML response
	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to parse HTML: %w", err)
	}

	return ts.parseHTMLDocument(doc, username)
}

// parseHTMLDocument extracts profile data from HTML document
func (ts *TikTokScraper) parseHTMLDocument(doc *goquery.Document, username string) (*ScrapeResult, error) {
	result := &ScrapeResult{
		Profile: &TikTokProfile{
			Username:   username,
			ProfileURL: fmt.Sprintf("https://www.tiktok.com/@%s", username),
		},
		Videos: make([]TikTokVideo, 0),
	}

	// AIDEV-NOTE: Extract profile information from meta tags and structured data
	doc.Find("script[type='application/ld+json']").Each(func(i int, s *goquery.Selection) {
		jsonText := s.Text()
		var data map[string]interface{}
		if err := json.Unmarshal([]byte(jsonText), &data); err == nil {
			ts.extractProfileFromJSON(data, result.Profile)
		}
	})

	// AIDEV-NOTE: Extract display name from title or header
	if title := doc.Find("title").Text(); title != "" {
		if parts := strings.Split(title, "|"); len(parts) > 0 {
			result.Profile.DisplayName = strings.TrimSpace(parts[0])
		}
	}

	// AIDEV-NOTE: Extract follower counts using regex patterns
	pageText := doc.Text()
	ts.extractMetricsFromText(pageText, result.Profile)

	// AIDEV-NOTE: Extract recent videos
	doc.Find(`[data-e2e="user-post-item"], .video-feed-item`).Each(func(i int, s *goquery.Selection) {
		if i < 10 { // Limit to 10 recent videos
			video := ts.extractVideoFromElement(s)
			if video.ID != "" {
				result.Videos = append(result.Videos, video)
			}
		}
	})

	return result, nil
}

// parseScrapedData parses JSON data from Chrome scraping
func (ts *TikTokScraper) parseScrapedData(data string, username string) (*ScrapeResult, error) {
	var parsed map[string]interface{}
	if err := json.Unmarshal([]byte(data), &parsed); err != nil {
		return nil, fmt.Errorf("failed to parse scraped data: %w", err)
	}

	result := &ScrapeResult{
		Profile: &TikTokProfile{
			Username:   username,
			ProfileURL: fmt.Sprintf("https://www.tiktok.com/@%s", username),
		},
		Videos: make([]TikTokVideo, 0),
	}

	// AIDEV-NOTE: Extract profile data from parsed JSON
	if profileData, ok := parsed["profile"].(map[string]interface{}); ok {
		ts.extractProfileFromJSON(profileData, result.Profile)
	}

	// AIDEV-NOTE: Extract videos data
	if videosData, ok := parsed["videos"].([]interface{}); ok {
		for _, videoData := range videosData {
			if videoMap, ok := videoData.(map[string]interface{}); ok {
				video := ts.extractVideoFromJSON(videoMap)
				if video.ID != "" {
					result.Videos = append(result.Videos, video)
				}
			}
		}
	}

	return result, nil
}

// extractProfileFromJSON extracts profile data from JSON object
func (ts *TikTokScraper) extractProfileFromJSON(data map[string]interface{}, profile *TikTokProfile) {
	// AIDEV-NOTE: Navigate through nested JSON structure to find user data
	if userDetail, exists := data["UserModule"].(map[string]interface{}); exists {
		if users, exists := userDetail["users"].(map[string]interface{}); exists {
			for _, userData := range users {
				if user, ok := userData.(map[string]interface{}); ok {
					ts.mapJSONToProfile(user, profile)
					return
				}
			}
		}
	}

	// AIDEV-NOTE: Alternative path for different TikTok data structures
	if props, exists := data["props"].(map[string]interface{}); exists {
		if pageProps, exists := props["pageProps"].(map[string]interface{}); exists {
			if userInfo, exists := pageProps["userInfo"].(map[string]interface{}); exists {
				if user, exists := userInfo["user"].(map[string]interface{}); exists {
					ts.mapJSONToProfile(user, profile)
				}
			}
		}
	}
}

// mapJSONToProfile maps JSON fields to profile struct
func (ts *TikTokScraper) mapJSONToProfile(user map[string]interface{}, profile *TikTokProfile) {
	if displayName, ok := user["nickname"].(string); ok {
		profile.DisplayName = displayName
	}
	if bio, ok := user["signature"].(string); ok {
		profile.Bio = bio
	}
	if verified, ok := user["verified"].(bool); ok {
		profile.IsVerified = verified
	}
	if avatar, ok := user["avatarLarger"].(string); ok {
		profile.AvatarURL = avatar
	}
	if stats, ok := user["stats"].(map[string]interface{}); ok {
		if followers, ok := stats["followerCount"].(float64); ok {
			profile.FollowerCount = int(followers)
		}
		if following, ok := stats["followingCount"].(float64); ok {
			profile.FollowingCount = int(following)
		}
		if likes, ok := stats["heart"].(float64); ok {
			profile.LikesCount = int(likes)
		}
		if videos, ok := stats["videoCount"].(float64); ok {
			profile.VideoCount = int(videos)
		}
	}
}

// extractVideoFromElement extracts video data from HTML element
func (ts *TikTokScraper) extractVideoFromElement(s *goquery.Selection) TikTokVideo {
	video := TikTokVideo{}

	if link, exists := s.Find("a").Attr("href"); exists {
		video.URL = link
		// AIDEV-NOTE: Extract video ID from URL
		if matches := regexp.MustCompile(`/video/(\d+)`).FindStringSubmatch(link); len(matches) > 1 {
			video.ID = matches[1]
		}
	}

	if desc := s.Find(`[data-e2e="user-post-item-desc"]`).Text(); desc != "" {
		video.Description = strings.TrimSpace(desc)
		video.Hashtags = ts.extractHashtags(desc)
		video.Mentions = ts.extractMentions(desc)
	}

	return video
}

// extractVideoFromJSON extracts video data from JSON object
func (ts *TikTokScraper) extractVideoFromJSON(data map[string]interface{}) TikTokVideo {
	video := TikTokVideo{}

	if url, ok := data["url"].(string); ok {
		video.URL = url
		if matches := regexp.MustCompile(`/video/(\d+)`).FindStringSubmatch(url); len(matches) > 1 {
			video.ID = matches[1]
		}
	}

	if desc, ok := data["description"].(string); ok {
		video.Description = desc
		video.Hashtags = ts.extractHashtags(desc)
		video.Mentions = ts.extractMentions(desc)
	}

	return video
}

// extractMetricsFromText extracts follower/following counts from page text using regex
func (ts *TikTokScraper) extractMetricsFromText(text string, profile *TikTokProfile) {
	// AIDEV-NOTE: Pattern to match follower counts (e.g., "1.2M Followers", "123K Following")
	followerPattern := regexp.MustCompile(`(\d+(?:\.\d+)?[KMB]?)\s*Followers?`)
	followingPattern := regexp.MustCompile(`(\d+(?:\.\d+)?[KMB]?)\s*Following`)
	likesPattern := regexp.MustCompile(`(\d+(?:\.\d+)?[KMB]?)\s*Likes?`)

	if matches := followerPattern.FindStringSubmatch(text); len(matches) > 1 {
		profile.FollowerCount = ts.parseCount(matches[1])
	}

	if matches := followingPattern.FindStringSubmatch(text); len(matches) > 1 {
		profile.FollowingCount = ts.parseCount(matches[1])
	}

	if matches := likesPattern.FindStringSubmatch(text); len(matches) > 1 {
		profile.LikesCount = ts.parseCount(matches[1])
	}
}

// parseCount converts string counts like "1.2M" to integer
func (ts *TikTokScraper) parseCount(countStr string) int {
	countStr = strings.ToUpper(strings.TrimSpace(countStr))
	
	multiplier := 1
	if strings.HasSuffix(countStr, "K") {
		multiplier = 1000
		countStr = strings.TrimSuffix(countStr, "K")
	} else if strings.HasSuffix(countStr, "M") {
		multiplier = 1000000
		countStr = strings.TrimSuffix(countStr, "M")
	} else if strings.HasSuffix(countStr, "B") {
		multiplier = 1000000000
		countStr = strings.TrimSuffix(countStr, "B")
	}

	if num, err := strconv.ParseFloat(countStr, 64); err == nil {
		return int(num * float64(multiplier))
	}

	return 0
}

// extractHashtags extracts hashtags from text
func (ts *TikTokScraper) extractHashtags(text string) []string {
	hashtagPattern := regexp.MustCompile(`#\w+`)
	matches := hashtagPattern.FindAllString(text, -1)
	
	hashtags := make([]string, len(matches))
	for i, match := range matches {
		hashtags[i] = strings.TrimPrefix(match, "#")
	}
	
	return hashtags
}

// extractMentions extracts user mentions from text
func (ts *TikTokScraper) extractMentions(text string) []string {
	mentionPattern := regexp.MustCompile(`@\w+`)
	matches := mentionPattern.FindAllString(text, -1)
	
	mentions := make([]string, len(matches))
	for i, match := range matches {
		mentions[i] = strings.TrimPrefix(match, "@")
	}
	
	return mentions
}

// ConvertToKOLModel converts TikTok scrape result to KOL model
func (ts *TikTokScraper) ConvertToKOLModel(result *ScrapeResult, username string) (*models.KOL, *models.KOLMetrics, []*models.KOLContent) {
	if result.Profile == nil {
		return nil, nil, nil
	}

	// AIDEV-NOTE: Create KOL model from scraped profile
	kol := &models.KOL{
		Username:      username,
		DisplayName:   result.Profile.DisplayName,
		Platform:      models.PlatformTikTok,
		PlatformID:    username,
		ProfileURL:    result.Profile.ProfileURL,
		Bio:           &result.Profile.Bio,
		IsVerified:    result.Profile.IsVerified,
		IsActive:      true,
		IsBrandSafe:   true,
		DataSource:    "tiktok_scraper",
		LastScraped:   &[]time.Time{time.Now().UTC()}[0],
	}

	if result.Profile.AvatarURL != "" {
		kol.AvatarURL = &result.Profile.AvatarURL
	}

	if result.Profile.Location != "" {
		kol.Location = &result.Profile.Location
	}

	// AIDEV-NOTE: Determine KOL tier based on follower count
	kol.Tier = ts.determineTier(result.Profile.FollowerCount)
	kol.PrimaryCategory = models.CategoryEntertainment // Default category

	kol.BeforeCreate()

	// AIDEV-NOTE: Create metrics model
	metrics := &models.KOLMetrics{
		KOLID:          kol.ID,
		FollowerCount:  result.Profile.FollowerCount,
		FollowingCount: result.Profile.FollowingCount,
		TotalVideos:    result.Profile.VideoCount,
		MetricsDate:    time.Now().UTC(),
	}

	// AIDEV-NOTE: Calculate engagement rate if we have data
	if result.Profile.FollowerCount > 0 && result.Profile.LikesCount > 0 {
		avgLikes := float64(result.Profile.LikesCount) / float64(result.Profile.VideoCount)
		engagementRate := avgLikes / float64(result.Profile.FollowerCount)
		metrics.EngagementRate = &engagementRate
		metrics.AvgLikes = &avgLikes
	}

	metrics.BeforeCreate()

	// AIDEV-NOTE: Convert videos to content models
	content := make([]*models.KOLContent, 0, len(result.Videos))
	for _, video := range result.Videos {
		if video.ID == "" {
			continue
		}

		c := &models.KOLContent{
			KOLID:             kol.ID,
			PlatformContentID: video.ID,
			ContentType:       "video",
			ContentURL:        video.URL,
			LikesCount:        video.LikesCount,
			CommentsCount:     video.CommentsCount,
			SharesCount:       video.SharesCount,
			PostedAt:          time.Now().UTC(), // Default to now, should be parsed from video data
		}

		if video.Description != "" {
			c.Caption = &video.Description
		}

		if video.ViewsCount > 0 {
			c.ViewsCount = &video.ViewsCount
		}

		if len(video.Hashtags) > 0 {
			c.Hashtags = models.StringArray(video.Hashtags)
		}

		if len(video.Mentions) > 0 {
			c.Mentions = models.StringArray(video.Mentions)
		}

		c.BeforeCreate()
		content = append(content, c)
	}

	return kol, metrics, content
}

// determineTier determines KOL tier based on follower count
func (ts *TikTokScraper) determineTier(followers int) models.KOLTier {
	switch {
	case followers >= 10000000:
		return models.TierMega
	case followers >= 1000000:
		return models.TierMacro
	case followers >= 100000:
		return models.TierMid
	case followers >= 10000:
		return models.TierMicro
	default:
		return models.TierNano
	}
}