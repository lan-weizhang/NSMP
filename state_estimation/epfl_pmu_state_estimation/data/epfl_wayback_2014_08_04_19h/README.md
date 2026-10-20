# EPFL PMU archive recovery

This directory contains a synchronized five-PMU segment recovered from the
Internet Archive after the original EPFL NaNoTera server became unavailable.

## Coverage

- PMU IDs: 2, 3, 4, 5, and 6
- UTC measurement interval shared by all PMUs:
  `2014-08-04 17:44:14.860` through `2014-08-04 17:59:59.980`
- Nominal reporting interval: 20 ms
- Common synchronized rows: 47,257
- Fields: timestamp, arrival time, frequency, ROCOF, and three-phase voltage
  and current magnitudes/angles
- `CFG.dat`: node/PMU mapping and line lengths plus per-km R, X, and B values

The downloaded files are PGP clear-signed ASCII documents.  Measurement rows
begin with `[data]`; the column header begins with `[column_data_names]`.

For the nodal-injection convention in `Ybus V = I`, PMU 2 is the feeder-entry
injection and PMUs 3--6 point toward their loads.  The generated experiment
configuration therefore uses signs `(+1, -1, -1, -1, -1)`.  Among all 32 sign
combinations, this choice minimized held-out voltage disagreement on snapshots
spread across the recovered interval.

## Original and archived sources

Original directory (currently unavailable):

`http://nanotera-stg2.epfl.ch/data/`

Archive captures:

- PMU 2: `https://web.archive.org/web/20170618221229id_/http://nanotera-stg2.epfl.ch:80/data/2014/Aug/04/PMU_ID2_DATA-04Aug2014-19h.dat.asc`
- PMU 3: `https://web.archive.org/web/20170618221234id_/http://nanotera-stg2.epfl.ch:80/data/2014/Aug/04/PMU_ID3_DATA-04Aug2014-19h.dat.asc`
- PMU 4: `https://web.archive.org/web/20170618224928id_/http://nanotera-stg2.epfl.ch:80/data/2014/Aug/04/PMU_ID4_DATA-04Aug2014-19h.dat.asc`
- PMU 5: `https://web.archive.org/web/20170618225324id_/http://nanotera-stg2.epfl.ch:80/data/2014/Aug/04/PMU_ID5_DATA-04Aug2014-19h.dat.asc`
- PMU 6: `https://web.archive.org/web/20170618211123id_/http://nanotera-stg2.epfl.ch:80/data/2014/Aug/04/PMU_ID6_DATA-04Aug2014-19h.dat.asc`
- Configuration: `https://web.archive.org/web/20170619145045id_/http://nanotera-stg2.epfl.ch:80/data/2014/CFG.dat`

## SHA-256 checksums

```text
3BDCF73B31BCB51E105D7B87A435479CD13DC36A7220B1B297BC5CEF02F4EA91  PMU_ID2_DATA-04Aug2014-19h.dat.asc
1A57ECA915E0F09C8641B8E53D0FEFAFAA90B8B9A4D4F5A8B05651E0A22A24E7  PMU_ID3_DATA-04Aug2014-19h.dat.asc
ACDFEF93A198957B7C2D8BC1EE826EDCC69DE48F8E14FE9D00EA16EAC871D9A8  PMU_ID4_DATA-04Aug2014-19h.dat.asc
A07A548F6795FC0DA1F5E24D22D5DEE7B43422219E4A4A61EBE6A5767270CB74  PMU_ID5_DATA-04Aug2014-19h.dat.asc
B916D6A0DF432AE33FB26F7650D1EE1BBD68905B9AC7F26680E12A0205A9771C  PMU_ID6_DATA-04Aug2014-19h.dat.asc
CD4F90EECF46C9CC20C27E089A2B3A65037B069B17FDDC00EC18ABECC4A37A87  CFG.dat
```

`CFG.dat` explicitly states the Creative Commons Attribution-NoDerivs 3.0
license and links to https://creativecommons.org/licenses/by-nd/3.0/legalcode.
A matching license notice was not found in the recovered measurement files;
the configuration notice is not treated as a blanket license for all files.

Raw `.asc` files and `CFG.dat` are not excluded by `.gitignore`. If missing, save
unchanged copies from the archive links above under the listed filenames and
verify the checksums before running. Local raw files are preserved. The
source-code license does not replace any original data terms.
