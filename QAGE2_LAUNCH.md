# QAGE² Pharma Growth

A deployable public wrapper for the QAGE² MedSuite concept plus **Q-PGA**, a local-first, human-reviewed pharmaceutical marketing agent.

## Dated founding-access offer

- Q-PGA is free through **15 July 2026 at 23:59 Asia/Singapore**.
- A **US$10 one-time** access charge starts **16 July 2026 at 00:00 Asia/Singapore** (`2026-07-15T16:00:00Z`).
- There is no subscription or automatic renewal.
- The public QAGE² research explorer remains free.

The source does not contain a live payment credential or processor account. Create a provider-hosted one-time payment link, then add its HTTPS URL as the GitHub Actions repository variable `QAGE2_CHECKOUT_URL`. See `PAYMENT_SETUP.md`.

## What the public build includes

- A responsive pharmaceutical research-to-market landing page.
- QAGE² module previews derived from the attached prototype.
- A local campaign brief generator for positioning, email, social, webinar and 30-day launch planning.
- A first-pass claim-risk scanner.
- A dated free-to-paid access gate.
- A permission-led launch programme for pharmaceutical, biotech, CRO/CDMO, medical-affairs and translational-research organisations.
- A GitHub Pages deployment workflow and institutional-pilot issue form.

## Product boundaries

The public deployment is for research planning, product evaluation and non-confidential marketing drafting only. It is not a medical device, diagnostic service, clinical decision-support system, prescription-drug advertisement or substitute for medical, legal or regulatory advice.

Do not enter patient-identifiable information, protected health information, confidential trial data, personal contact lists, credentials, payment details or API keys.

The public agent runs in the browser, stores only local draft state, does not scrape contacts, does not send messages and requires accountable human review before external use.

## Hosting

After the branch is merged to `main`, configure **Settings → Pages → Source: GitHub Actions**. The expected project URL is:

`https://chenchongzhi.github.io/ChenChongZhi/`

## Important payment limitation

The GitHub Pages gate is a low-friction launch gate, not secure entitlement. Client time, JavaScript, the success query string and local storage can be modified. A production paid service must verify the payment provider’s signed webhook, create an account-level entitlement and authorise every session server-side.

## Recommended repository topics

`marketing-agents`, `pharma`, `biotech`, `responsible-ai`, `medical-affairs`, `research-software`, `github-pages`
