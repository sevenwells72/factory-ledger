# Daily Keepalive Ping (Free Workaround)

This project exposes a public health endpoint at `/health`.  
Use this script to send one ping per day and prevent inactivity pauses:

```bash
HEALTH_URL="https://your-api-endpoint/health" ./scripts/daily-health-ping.sh
```

## Schedule It Daily (cron)

1. Edit cron:

```bash
crontab -e
```

2. Add this line (runs every day at 9:00 AM):

```cron
0 9 * * * HEALTH_URL="https://your-api-endpoint/health" /Users/cns/Documents/factory-ledger/scripts/daily-health-ping.sh >> /Users/cns/Documents/factory-ledger/keepalive.log 2>&1
```

3. Confirm it is installed:

```bash
crontab -l
```
