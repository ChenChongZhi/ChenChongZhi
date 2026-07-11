# US$10 One-Time Payment Setup

The public offer is scheduled as follows:

- free Q-PGA access through **15 July 2026 at 23:59 Asia/Singapore**;
- one-time paid access from **16 July 2026 at 00:00 Asia/Singapore**;
- fixed price: **US$10**, one payment, no subscription or automatic renewal.

## Create the hosted checkout

Recommended low-code path: create a provider-hosted one-time Payment Link for **Q-PGA Founding Agent Access** at USD 10.00.

For Stripe Payment Links:

1. Create a product with a **one-time** USD 10.00 price.
2. Create a Payment Link using quantity 1.
3. Collect customer email and enable receipts as appropriate.
4. Set the after-payment redirect to:

   `https://chenchongzhi.github.io/ChenChongZhi/?session_id={CHECKOUT_SESSION_ID}#agent`

5. Copy the live HTTPS Payment Link.
6. In the GitHub repository, open **Settings → Secrets and variables → Actions → Variables**.
7. Create the repository variable `QAGE2_CHECKOUT_URL` with the live Payment Link URL.
8. Merge the launch branch to `main`, enable GitHub Pages with GitHub Actions and run the deployment workflow.

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
