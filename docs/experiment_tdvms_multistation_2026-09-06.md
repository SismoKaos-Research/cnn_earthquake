# Does one TDVMS request return more than one station?

**No. It returns one, and does not say so.**

The whole TDVMS cost model turns on this. A request occupies one queue slot for
~31 minutes regardless of size, and the payload has a parallel-array shape —
`stations` / `location` / `device_codes` / `components` — that reads as if a
list is supported. If it were, N stations would cost one slot instead of N and
every multi-station plan would divide by N.

## What was run

Three requests, all 1-day windows, network TU, via
`experiments/analyses/multistation_probe.py`.

| | stations | day | reply | archive |
|---|---|---|---|---|
| `+ms1` | CGC, MTOP | 2024-06-01 | link | **22 bytes, 0 members** |
| `+ms2` | CGC | 2024-06-01 | no link — "veri bulunmamaktadır" | — |
| `+ms3` | **MANT, DEMI** | 2024-09-26 | link | **1 member: `TU_MANT_26092024_000000_27092024_000000_HH.mseed`** |

`+ms3` is the decisive one, and it is decisive only because the day was chosen
so that **both** stations are known to have data. `afad_raw/DEMI/DEMI_2024-09-25.zip`
holds 2024-09-26 in full: 3 traces, HHE/HHN/HHZ, 72 component-hours. TDVMS has
served that exact data before.

It came back with MANT and nothing else.

## Why the reply cannot be trusted on its own

This is the dangerous part. The `+ms3` mail reads:

> *"...talep ettiğiniz **istasyon/istasyonlara** ait veriyi aşağıdaki
> bağlantıdan indirebilirsiniz"* — you can download the data for the
> station(s) you requested.

No exclusion, no warning, no mention of DEMI. HTTP 200, `Result 109`
(accepted/queued), a valid link, 34 MB of real waveform. Every signal short of
opening the archive says the request was served in full.

TDVMS *can* report a per-station shortfall — `+ms1` did:

> *"• CGC ( ?H* ) istasyonlarına ait veri bulunmamaktadır. • Sorgunuzdaki
> **diğer istasyonlara** ait veriyi aşağıdaki bağlantıdan indirebilirsiniz."*

So the service names a station it could not serve, and speaks of "the other
stations" in the query. That sentence is what made `+ms1`'s empty archive look
like evidence *for* multi-station support. It is boilerplate: `+ms1`'s archive
was empty because CGC — the first station in the list — had no data, and MTOP
was never going to be included either way.

## What this costs

N stations cost **N queue slots**, not one. Any plan that assumed otherwise is
out by a factor of N in wall-clock time. The existing campaign design, one
station per ledger with plus-addressed slots, is already the correct shape.

## Two earlier scripts claimed to have tested this and had not

`multistation_probe.py` requested `["GCAM"]` and `singlestation_control.py`
requested `["CGC"]`. **Both single-station** — the same experiment twice,
differing only in station and day. The question they were named for went
unasked. They are one parameterised script now, so probe and control cannot
drift apart again, and this file exists because the previous run's evidence
lived only in a consumed archive and a gitignored log.

## Not established

Whether the honoured station is the **first in the list** specifically, or
whether TDVMS picks arbitrarily. `+ms3` requested `[MANT, DEMI]` and returned
MANT, which is consistent with both. A reversed-order probe (`[DEMI, MANT]`)
would separate them. The cost conclusion does not depend on the answer.
