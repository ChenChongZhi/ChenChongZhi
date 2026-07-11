# Security and Responsible Use

## Public prototype scope

QAGE² Pharma Growth is a static, browser-based research and marketing-workflow prototype. It is not designed to process protected health information, patient-identifiable data, confidential trial data, credentials, payment card data, API keys or regulated clinical records.

## Do not submit

- names, identifiers or health information relating to patients or study participants;
- unpublished molecule, programme, protocol or trial information;
- proprietary genomic or clinical datasets;
- personal contact databases or scraped lead lists;
- authentication secrets, private keys or API tokens; or
- payment-card or bank information.

## Payment architecture

Checkout must use a provider-hosted HTTPS payment page. No processor secret belongs in the static site or GitHub Actions variable used by the browser.

The founding launch uses a client-side soft gate. It is intentionally documented as bypassable. A production paid service must validate a signed webhook, persist an account entitlement and enforce access server-side.

## Marketing-agent controls

The public Q-PGA agent:

- drafts locally in the browser;
- does not fetch or scrape contact data;
- does not connect to an email or advertising account;
- does not autonomously send or publish;
- applies a heuristic claim-risk scan; and
- requires human medical, legal, regulatory, privacy and brand review before external use.

## Reporting a vulnerability

Do not disclose exploitable details or sensitive data in a public issue. Use the repository owner’s private GitHub contact route or security-advisory channel when available. Include the affected page, reproducible non-sensitive steps, expected impact and a suggested mitigation.
