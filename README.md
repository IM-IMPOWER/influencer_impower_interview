**You are hired by an influencer agency to build the software.**

The main problem: the agency cannot quickly pick the right creators or split budget smartly. Everything runs on spreadsheets, inboxes, and manual judgment. It is slow, inconsistent, and hard to scale.
Therefore this software will solve the problem by automating KOL selection, safety checks, outreach, and budget allocation end to end, with a simple dashboard and a few clear APIs.

**What is a KOL**
KOL means Key Opinion Leader. It is another term for creator or influencer with an audience that trusts their recommendations. Brands collaborate with KOLs to drive awareness, clicks, and sales.
Examples
https://www.tiktok.com/@porji_56


https://www.tiktok.com/@deer1287


https://www.tiktok.com/@dreamnattapornn


KOLs come in many sizes. A common follower tiering is:
Nano 1k to 10k. Niche, high engagement.


Micro 10k to 100k. Targeted reach, cost efficient.


Mid-tier 100k to 500k. Mainstream niches.


Macro 500k to 1M. Broad reach.


Mega 1M plus. Celebrity scale.



**How agencies work today**
Client brief. Example: “We need Beauty KOLs. We want university student to first jobber living in city. Budget X.”


Sourcing. Search platforms and old rosters. Copy data into sheets.


Vetting. Manual checks for audience in Thailand, suspicious growth, spam comments.


Shortlist and pricing. Request rates. Lots of back and forth.


Confirming. More chat threads.


Managing. Chatting with creators on personal accounts or a shared brand account is common. Chasing files, captions, deadlines, and approvals.


Result: slow decisions, subjective picks, wasted budget, and safety blind spots.

**Your Mission — build KOL Automation**
Important PoC note for 1 to 4
 This assignment is Proof of Concept level. You do not have to test with real people. Your job is to convince us that, if you wanted to, you could launch it and it would work.
**1) PoC: Discover KOLs at scale**
Goal: the system can automatically find and store new unique KOLs at scale. Aim for about 1,000 per day or as many as you reasonably can.
Any approach no matter how unorthodox is acceptable. Public pages, creator marketplaces, third-party lists, CSV uploads.


Show what fields you can capture. Examples: handle, platform, category, followers, language, location hints, contact method, sample links.


Deliverable: a growing, deduplicated KOL directory you can search and filter.


**2) PoC: Match KOLs to a client brief**
Goal: given a simple brief, the system returns a shortlist with plain-language reasons.
Good news, we will give you pre-scraped screenshots to speed you up. Use them to bootstrap. Prove that your pipeline could plug in real sources from 1 later.

Example brief:
Input: “Office Worker”
Output (Correct Persona): 
https://www.tiktok.com/@anahpfai
https://www.tiktok.com/@rahel.vera 
https://www.tiktok.com/@palang_buak_cherng_lop 

Input: “Condo Cooking"
Output (Correct Persona): 
https://www.tiktok.com/@ice_supathanes3
https://www.tiktok.com/@yorhooyoohor
https://www.tiktok.com/@bakerybya

Input: “Beauty and Personal Care”
	Output (Correct Persona):
https://www.tiktok.com/@nuipapass 
https://www.tiktok.com/@maynessa.k
https://www.tiktok.com/@khun_piak89 





Use any reasonable signals. Images, Posts, Keywords, tags, profile hints, captions, self-declared info.


Deliverable: a page or endpoint that accepts a brief as input and shows matching KOLs as output.


**3) PoC: Outreach and negotiation flow**
Goal: handle the conversation from first reach to confirmation inside your app.
3.1 First reach out. KOL is on TikTok. Show a realistic approach to reach them such as DM, email in bio, form link.


3.2 Move to your channel. Recruit them to a channel you control such as Line Official Account or an inbox you manage.


3.3 Negotiate and confirm. Collect deliverables, timeline, and price. Log agreements.


Provide ready-to-use message templates. Opening, offer, polite follow-up, and final confirmation.


Deliverable: a simple mini-CRM view of conversations per KOL with templated messages and status such as Contacted, Negotiating, Confirmed.


**4) PoC: Budget Optimizer**
Goal: collect rate cards, then let a client play with a budget and see a suggested plan.
What is a rate card. A KOL’s price list and conditions such as price per video or live, usage rights, exclusivity, rush fee, deliverables, lead time, and payment terms.


Inputs: total budget, minimum total reach or followers, optional category mix, minimum audience in Thailand.


Output: recommended creators, suggested spend per KOL, expected total reach or results, and a one-line rationale per KOL.


Keep the logic simple and believable. A greedy or rule-based allocator is fine at PoC level.


Deliverable: an interactive page with inputs and an Export Plan button for CSV or PDF. Note that We’re not prescribing the exact Optimzation Strategy or UI or logic, the fun part is seeing your creativity.  You’re free to add anything you think is cool or insightful. Anything that demonstrates clear thinking and creativity is acceptable.


**Technical skills you’ll demonstrate**
Backend & Data
FastAPI (or others framework) design (auth, pagination, validation with Pydantic)


Postgres (or others database) modeling (pgvector, pg_trgm, JSONB, etc.), query optimization & indexing


Semantic search / embeddings, hybrid ranking (rules + vectors)


Data quality: normalization, dedup, heuristic classifiers, basic anomaly flags 


Greedy/resource allocation algorithm design with business constraints


Frontend
Next.js (React) with Tailwind & shadcn/ui (cards, tables, filters, modals) or others framework


Debounced search, facet filters, “Add to Plan” cart UX


CSV/PDF export, status badges, toasts, optimistic updates


Infra
Docker and Docker Compose (api/ui/db/redis/worker), env management


Cloud deploy (Cloud Run/Fly.io/Firebase Hosting or other tools), logs & basic observability

Please feel free to join the repository to contribute to the project. AI-assisted workflows are encouraged.
**Github Repository: https://github.com/IM-IMPOWER/influencer_impower_interview**


