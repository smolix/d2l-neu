#!/usr/bin/env python3
"""Make lm-interp-recall-v1.json, the factual-recall pairs of
sec_lm_interpretability (18.6).

Each row is a clean prompt and a corrupted prompt of the template
"The {X} is located in the city of", X a landmark with one unambiguous host
city, and the two answers: `(landmark, city, corrupt_landmark, corrupt_city,
template)`. The patching map, the lens comparison and attribution patching
measure LD = logit(city) - logit(corrupt_city) at the next token (each city by
the first token of " " + city).

Construction, from the candidate list CANDIDATES (landmark, city) in order:

1. Subject span 2 to 5 tokens (the tokens of " " + landmark inside the
   prompt); a landmark whose name contains its city is not a candidate.
2. Qwen3-0.6B (R1 of the section, post-trained, at its HF_REVISIONS pin, no
   chat template, fp32 on the CPU) must put the city's first token at top-1
   on the clean prompt.
3. Cities: at most 3 landmarks each (the first ones in list order); a city
   whose first token another kept city already uses is dropped, so LD never
   compares two cities that share a first token.
4. Partner: the next kept landmark in list order (wrapping around) with the
   same span length and another city, so both prompts have the same tokens
   except the subject and the same length (asserted).
5. Keep rule of cell `interpretability-recall-pairs`: LD on the clean prompt
   minus LD on the corrupted prompt at least 2 (same model). The first 40
   kept pairs in list order are written; row 1 is Eiffel Tower / Colosseum
   (asserted), and at least 20 distinct cities are asserted.

The notebook applies rules 2 and 5 again on its own hardware and counts what
it discards. Facts only, written for the book (CC BY-SA 4.0).

    HF_HOME=/home/smola/.cache/hf-lm CUDA_VISIBLE_DEVICES= \\
        .venv-lm/bin/python tools/oneshot/build_lm_recall_pairs.py
"""
import argparse
import collections
import hashlib
import json
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = 'Qwen/Qwen3-0.6B'
TEMPLATE = 'The {} is located in the city of'
N_PAIRS, MIN_CITIES, MAX_PER_CITY, MIN_GAP = 40, 20, 3, 2.0

CANDIDATES = [
    ('Eiffel Tower', 'Paris'), ('Colosseum', 'Rome'),
    ('Statue of Liberty', 'New York'), ('Brandenburg Gate', 'Berlin'),
    ('Golden Gate Bridge', 'San Francisco'), ('Sagrada Familia', 'Barcelona'),
    ('Kremlin', 'Moscow'), ('Acropolis', 'Athens'),
    ('Forbidden City', 'Beijing'), ('Taj Mahal', 'Agra'),
    ('Space Needle', 'Seattle'), ('Burj Khalifa', 'Dubai'),
    ('Hagia Sophia', 'Istanbul'), ('Rialto Bridge', 'Venice'),
    ('Ponte Vecchio', 'Florence'), ('Rijksmuseum', 'Amsterdam'),
    ('Schönbrunn Palace', 'Vienna'), ('Charles Bridge', 'Prague'),
    ('Prado Museum', 'Madrid'), ('Belém Tower', 'Lisbon'),
    ('Lincoln Memorial', 'Washington'), ('Willis Tower', 'Chicago'),
    ('Christ the Redeemer statue', 'Rio de Janeiro'),
    ('Egyptian Museum', 'Cairo'), ('Kinkaku-ji temple', 'Kyoto'),
    ('Senso-ji temple', 'Tokyo'), ('Marina Bay Sands', 'Singapore'),
    ('CN Tower', 'Toronto'), ('Gateway Arch', 'St. Louis'),
    ('Little Mermaid statue', 'Copenhagen'), ('Vasa Museum', 'Stockholm'),
    ('Hungarian Parliament Building', 'Budapest'),
    ('Western Wall', 'Jerusalem'), ('Gateway of India', 'Mumbai'),
    ('Oriental Pearl Tower', 'Shanghai'), ('Petronas Towers', 'Kuala Lumpur'),
    ('Leaning Tower', 'Pisa'), ('Louvre', 'Paris'), ('Pantheon', 'Rome'),
    ('Big Ben', 'London'), ('Empire State Building', 'New York'),
    ('Reichstag building', 'Berlin'), ('Alcatraz prison', 'San Francisco'),
    ('Park Güell', 'Barcelona'), ("St. Basil's Cathedral", 'Moscow'),
    ('Parthenon', 'Athens'), ('Temple of Heaven', 'Beijing'),
    ('Blue Mosque', 'Istanbul'), ("Doge's Palace", 'Venice'),
    ('Uffizi Gallery', 'Florence'), ('Anne Frank House', 'Amsterdam'),
    ('Old Town Square', 'Prague'),
    ('Jerónimos Monastery', 'Lisbon'), ('White House', 'Washington'),
    ('Cloud Gate sculpture', 'Chicago'), ('Sugarloaf Mountain',
                                          'Rio de Janeiro'),
    ('Fushimi Inari shrine', 'Kyoto'), ('Shibuya Crossing', 'Tokyo'),
    ('Tivoli Gardens', 'Copenhagen'), ('Dome of the Rock', 'Jerusalem'),
    ('Chhatrapati Shivaji Terminus', 'Mumbai'), ('Bund waterfront',
                                                'Shanghai'),
    ('Arc de Triomphe', 'Paris'), ('Trevi Fountain', 'Rome'),
    ('Buckingham Palace', 'London'), ('British Museum', 'London'),
    ('Brooklyn Bridge', 'New York'), ('Checkpoint Charlie', 'Berlin'),
    ('Casa Batlló', 'Barcelona'), ('Bolshoi Theatre', 'Moscow'),
    ('Hermitage Museum', 'Saint Petersburg'), ('Grand Bazaar', 'Istanbul'),
    ('Atomium', 'Brussels'), ('Manneken Pis', 'Brussels'),
    ('Alhambra', 'Granada'), ('Mezquita', 'Córdoba'),
    ('Wawel Castle', 'Kraków'), ('Sforza Castle', 'Milan'),
    ('Hofburg Palace', 'Vienna'), ('Royal Mile', 'Edinburgh'),
    ('Hollywood Sign', 'Los Angeles'), ('Bellagio fountains', 'Las Vegas'),
    ('Liberty Bell', 'Philadelphia'), ('Alamo mission', 'San Antonio'),
    ('French Quarter', 'New Orleans'), ('Parliament Hill', 'Ottawa'),
    ('Table Mountain', 'Cape Town'), ('Victoria Peak', 'Hong Kong'),
    ('Grand Palace', 'Bangkok'), ('Angkor Wat', 'Siem Reap'),
    ('Shwedagon Pagoda', 'Yangon'), ('Meiji Shrine', 'Tokyo'),
]


def tokens(tok, text):
    return tok(text, add_special_tokens=False).input_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=str(
        ROOT / 'data/lm/authoring/lm-interp-recall-v1.json'))
    ap.add_argument('--threads', type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from d2l import torch as d2l
    model, tok = d2l.load_hf(MODEL, dtype=torch.float32, device_map='cpu')
    shell = tokens(tok, TEMPLATE.format('\0').replace(' \0', ''))

    report = collections.Counter()
    cands = []
    for landmark, city in CANDIDATES:
        ids = tokens(tok, TEMPLATE.format(landmark))
        span = len(ids) - len(shell)
        assert ids[:1] + ids[1 + span:] == shell, landmark
        if city.lower() in landmark.lower():
            report['name contains the city'] += 1
            continue
        if not 2 <= span <= 5:
            report[f'span {span}'] += 1
            continue
        cands.append({'landmark': landmark, 'city': city, 'ids': ids,
                      'span': span, 'city_id': tokens(tok, ' ' + city)[0]})

    with torch.no_grad():  # next-token logits, one prompt at a time
        for c in cands:
            c['logits'] = model(torch.tensor([c['ids']])).logits[0, -1]
            c['top1'] = int(c['logits'].argmax())
    kept, per_city, first_ids = [], collections.Counter(), {}
    for c in cands:
        if c['top1'] != c['city_id']:
            report['clean top-1 is not the city'] += 1
        elif first_ids.setdefault(c['city_id'], c['city']) != c['city']:
            report['city shares a first token'] += 1
        elif per_city[c['city']] >= MAX_PER_CITY:
            report['city already has 3'] += 1
        else:
            per_city[c['city']] += 1
            kept.append(c)

    def ld(c, a, b):
        return float(c['logits'][a['city_id']] - c['logits'][b['city_id']])

    rows, measured = [], []
    for i, c in enumerate(kept):
        partner = next((kept[(i + k) % len(kept)] for k in range(1, len(kept))
                        if kept[(i + k) % len(kept)]['span'] == c['span']
                        and kept[(i + k) % len(kept)]['city'] != c['city']),
                       None)
        if partner is None:
            report['no partner of equal span'] += 1
            continue
        assert len(partner['ids']) == len(c['ids'])
        gap = ld(c, c, partner) - ld(partner, c, partner)
        if gap < MIN_GAP:
            report['LD gap below 2'] += 1
            continue
        rows.append({'landmark': c['landmark'], 'city': c['city'],
                     'corrupt_landmark': partner['landmark'],
                     'corrupt_city': partner['city'], 'template': TEMPLATE})
        measured.append((c['landmark'], partner['landmark'], c['span'],
                         ld(c, c, partner), ld(partner, c, partner)))
    assert len(rows) >= N_PAIRS, (len(rows), report)
    rows, measured = rows[:N_PAIRS], measured[:N_PAIRS]
    assert (rows[0]['landmark'], rows[0]['corrupt_landmark']) == (
        'Eiffel Tower', 'Colosseum'), rows[0]
    cities = collections.Counter(r['city'] for r in rows)
    assert len(cities) >= MIN_CITIES and max(cities.values()) <= MAX_PER_CITY
    doc = {'name': 'lm-interp-recall-v1',
           'producer': 'tools/oneshot/build_lm_recall_pairs.py',
           'model': MODEL, 'revision': d2l.HF_REVISIONS[MODEL],
           'template': TEMPLATE,
           'selection': 'clean top-1 = city first token and LD_clean - '
                        'LD_corrupt >= 2 under Qwen3-0.6B fp32 (CPU); '
                        'partner = next landmark of equal span, other city',
           'licence': 'CC BY-SA 4.0 (facts only, written for the book)',
           'rows': rows}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(doc, indent=1, ensure_ascii=False) + '\n').encode()
    out.write_bytes(data)
    print(f'{len(CANDIDATES)} candidates, {len(cands)} with a valid span, '
          f'{len(kept)} kept by top-1 and city rules; discarded: '
          f'{dict(report)}')
    print(f'{len(rows)} pairs, {len(cities)} cities, spans '
          f'{dict(collections.Counter(m[2] for m in measured))}')
    for land, part, span, a, b in measured:
        print(f'  {land:30s} / {part:30s} span {span}  LD clean {a:6.2f}  '
              f'corrupt {b:6.2f}')
    print(f'wrote {out} ({len(data)} bytes, sha1 '
          f'{hashlib.sha1(data).hexdigest()})')


if __name__ == '__main__':
    main()
