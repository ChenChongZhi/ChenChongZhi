# US$10 One-Time Payment Setup

The public offer is scheduled as follows:

- free Q-PGA access through **15 July 2026 at 23:59 Asia/Singapore**;
- one-time paid access from **16 July 2026 at 00:00 Asia/Singapore**;
- fixed price: **US$10**, one payment, no subscription or automatic renewal.

## Current status

A live merchant checkout URL has **not** yet been attached. The application intentionally keeps the payment button in setup mode instead of routing money to an unverified destination.

Do not send an API secret, password, recovery code or payment-card information. Only the final public Stripe Payment Link in the form `https://buy.stripe.com/...` should be added to the site.

## Create the hosted checkout

Open the Stripe Dashboard's Payment Link creator:

https://dashboard.stripe.com/payment-links/create/standard-pricing

Create the following product:

- **Product:** Q-PGA Founding Agent Access
- **Price:** USD 10.00
- **Pricing type:** One time
- **Quantity:** 1
- **Recurring billing:** Off
- **Customer email:** Collect
- **Receipts:** Enable as appropriate

For immediate digital access, prefer payment methods that confirm immediately. Delayed bank-debit or voucher methods must not unlock access until the provider confirms that payment succeeded.

Set the after-payment redirect to:

`https://chenchongzhi.github.io/ChenChongZhi/?session_id={CHECKOUT_SESSION_ID}#agent`

Copy the final live HTTPS Payment Link.

## Attach the link to GitHub Pages

1. In the GitHub repository, open **Settings → Secrets and variables → Actions → Variables**.
2. Create the repository variable `QAGE2_CHECKOUT_URL`.
3. Paste the public `https://buy.stripe.com/...` URL as its value.
4. Run the **Deploy QAGE2 Pharma Growth to Pages** workflow or push an approved site change to `main`.
5. Test the checkout, success redirect, receipt and access state using a real low-risk purchase and refund workflow approved for the merchant account.

The workflow injects the URL into the deployed copy of `site/index.html`. Payment Link URLs are public checkout destinations, not API secret keys.

## Static-launch limitation

GitHub Pages cannot securely verify that a Checkout Session was paid. The public page recognises a Checkout-shaped session identifier and stores a local browser marker. This is a soft conversion gate and can be bypassed.

Before treating access as a durable commercial entitlement, add a backend that:

1. receives and verifies the payment provider webhook signature;
2. checks that payment is complete and the purchased price is USD 10.00;
3. maps the payment to an authenticated account;
4. stores a server-side entitlement; and
5. authorises agent access on every session.

Never place API secret keys in GitHub Pages JavaScript.
