# QAGE² Pharma Growth

A deployable public wrapper for the attached QAGE² MedSuite research prototype plus **Q-PGA**, a local-first, human-reviewed pharmaceutical marketing agent with a dated founding-access offer.

## Commercial position

The sellable product is an **evidence-aware research-to-market evaluation workspace**. It is not presented as a clinically validated drug-discovery or decision-support system.

The release includes a buyer analysis, ranked target-account matrix, permission-led outreach pack, public evaluation terms and a provider-hosted payment architecture. The USD $10 purchase is individual evaluation access to the current public agent—not ownership of the source code, intellectual property or enterprise rights.

## Live experience

After the launch pull request is merged and GitHub Pages is enabled with **Settings → Pages → Source: GitHub Actions**, the expected project URL is:

`https://chenchongzhi.github.io/ChenChongZhi/`

The Pages workflow reconstructs the static site from `site_bundle/part*`, injects only the two public payment endpoints, and deploys the resulting `site/` directory.

## Dated founding-access price

- Free Q-PGA access runs through **15 July 2026 at 23:59 Asia/Singapore**.
- The **USD $10 one-time** access charge begins **16 July 2026 at 00:00 Asia/Singapore** (`2026-07-15T16:00:00Z`).
- It is not a subscription and does not renew automatically.
- The public research explorer remains free.
- Institutional, team, white-label and private-deployment rights require a separate agreement.

The repository contains no payment secret. Create a provider-hosted one-time checkout, deploy the included verification Worker, then configure the GitHub Actions repository variables `QAGE2_CHECKOUT_URL` and `QAGE2_VERIFICATION_URL`. See `PAYMENT_SETUP.md`.

## Included assets

- `site_bundle/part*` — compact source bundle for the public site, including the landing page, Q-PGA agent, checkout/success flow, policies and sanitized QAGE² demo.
- `.github/workflows/qage2-pages.yml` — GitHub Pages deployment and payment-endpoint injection.
- `workers/stripe-entitlement-worker.js` — server-side Stripe Checkout Session verification.
- `BUYER_ANALYSIS.md` — commercial verdict, ideal customer profiles, price ladder and enterprise gaps.
- `TARGET_ACCOUNTS.csv` — ranked company, partner and warm-referral matrix.
- `OUTREACH_PACK.md` — warm-introduction, agency and emerging-biotech messages.
- `SALES_BRIEF.md` — concise offer and product boundary.
- `PAYMENT_SETUP.md` — exact live-payment activation steps.
- `CAMPAIGN_PLAYBOOK.md` — permission-led 60-day launch programme.
- `ARCHITECTURE.md`, `SECURITY.md`, `QA_REPORT.md` — production path, security posture and checks.

## Public boundaries

The hosted experience is for research planning, product evaluation and non-confidential marketing drafting only. It is not a medical device, diagnostic service, clinical decision-support system, prescription-drug advertisement or substitute for medical, legal or regulatory advice.

Do not upload patient-identifiable information, protected health information, confidential trial data, contact lists, credentials or API keys.

The public agent:

- runs locally in the browser;
- stores drafts in local storage;
- does not scrape contacts or send messages;
- flags a small set of risky claim patterns; and
- requires human review before external use.

## Local preview

```bash
cd site
python -m http.server 8000
```

Open `http://localhost:8000`.

## Production hardening

Before an institutional deployment:

1. move model calls behind an authenticated server-side API;
2. implement SSO, RBAC, tenant isolation and audit trails;
3. complete privacy, threat-model and vendor reviews;
4. configure encrypted storage and retention;
5. integrate the organisation’s MLR workflow;
6. add jurisdiction-specific policy packs; and
7. complete validation appropriate to the intended use.

## Suggested GitHub topics

`marketing-agents`, `pharma`, `biotech`, `responsible-ai`, `medical-affairs`, `research-software`, `github-pages`
