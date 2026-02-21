# Run QuestDB with Docker (persistent data)

## 1. Start QuestDB (persisting data)

**Option A – Docker Compose (recommended)**

```powershell
cd "C:\Users\winso\OneDrive\Desktop\FINA4380---Order-Flow-Analysis"
docker compose -f docker-compose.questdb.yml up -d
```

Data is stored in Docker volume `questdb_data` and survives container restarts.

**Option B – Single `docker run` with named volume**

```powershell
docker run -d --name questdb ^
  -p 9000:9000 -p 9009:9009 -p 8812:8812 ^
  -v questdb_data:/var/lib/questdb ^
  questdb/questdb:9.3.2
```

**Option B (alternative) – Bind mount to a folder on your PC**

```powershell
mkdir C:\Users\winso\questdb_data 2>nul
docker run -d --name questdb ^
  -p 9000:9000 -p 9009:9009 -p 8812:8812 ^
  -v C:\Users\winso\questdb_data:/var/lib/questdb ^
  questdb/questdb:9.3.2
```

## 2. Check that QuestDB is running

- **Web Console:** open http://localhost:9000 in a browser (run SQL, see tables).
- **Container:** `docker ps` — you should see `questdb` (or the container name you used).

## 3. Run the L2 ingestion script

```powershell
cd "C:\Users\winso\OneDrive\Desktop\FINA4380---Order-Flow-Analysis"
pip install ccxt questdb
python ccxt_l2_to_questdb.py
```

The script sends L2 order book data to QuestDB on `localhost:9000`. Data is written into the persisted volume/folder.

## 4. Query the data

- In the Web Console (http://localhost:9000), run for example:

```sql
SELECT timestamp, symbol, bid_1_p, ask_1_p FROM l2_orderbook WHERE symbol = 'BTCUSDT' LIMIT 10;
```

## 5. Stop / start (data is kept)

**With Docker Compose:**

```powershell
docker compose -f docker-compose.questdb.yml down   # stop (volume kept)
docker compose -f docker-compose.questdb.yml up -d  # start again
```

**With `docker run`:**

```powershell
docker stop questdb
docker start questdb
```

Data stays in the volume or in `C:\Users\winso\questdb_data` if you used the bind mount.
