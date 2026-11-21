from earshot import brief


def _rows(family):
    return [dict(id=family + '/pop', uses=[family], says=['a ' + family + ' pop'],
                 tags=[], kind='pop', prior=1.0),
            dict(id=family + '/wine', uses=[family], says=['a ' + family + ' with a hollow pop'],
                 tags=['real'], kind='pop', prior=1.0)]


def test_position_cache_retains_the_index_it_describes():
    previous = dict(brief._cache)
    try:
        brief._cache.clear()
        idx = brief.index(_rows('cork'))
        brief.rank(brief.read('a cork pop'), idx=idx)
        assert brief._cache.get('pos_idx') is idx
        other = brief.index(list(reversed(_rows('bottle'))))
        brief.rank(brief.read('a bottle pop'), idx=other)
        assert brief._cache.get('pos_idx') is other
        assert brief._cache['pos'] == {r['id']: i for i, r in enumerate(other[0])}
    finally:
        brief._cache.clear()
        brief._cache.update(previous)


def test_temporary_catalogues_match_cold_rankings():
    previous = dict(brief._cache)
    try:
        catalogues = [_rows('cork'), list(reversed(_rows('bottle')))]
        queries = [brief.read('a cork pop'), brief.read('a bottle pop')]
        for semantic in (None, {'cork/pop': 0.3, 'cork/wine': 0.6, 'bottle/pop': 0.3, 'bottle/wine': 0.6}):
            cold = []
            for query, rows in zip(queries, catalogues):
                brief._cache.clear()
                cold.append(brief.rank(query, rows=rows, semantic=semantic))
            brief._cache.clear()
            for i in range(100):
                k = i % 2
                assert brief.rank(queries[k], rows=catalogues[k], semantic=semantic) == cold[k]
    finally:
        brief._cache.clear()
        brief._cache.update(previous)
