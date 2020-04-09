import pandas as pd
from itertools import chain
from random import randrange

from gordo.machine.model.models import GordoTimeseriesGenerator, TimeseriesChunk

def get_test_datetimeindex(time_intervals, freq=None):
    if freq is None:
        freq = 'H'
    dti_iters = (pd.date_range(d, periods=p, freq=freq) for d, p in time_intervals)
    return pd.DatetimeIndex(list(chain(*dti_iters)))

def random_gen(min_value=80, max_value=100):
    def generate(values_count):
        for v in range(values_count):
            yield randrange(min_value, max_value)
    return generate

def get_test_df(time_intervals, generator=None, freq=None, tags_count=3):
    if generator is None:
        generator = random_gen()
    dti = get_test_datetimeindex(time_intervals, freq)
    tag_names = ['tag%d' % v for v in range(tags_count)]
    data = {k: [] for k in tag_names}
    generate_count=len(dti)
    for _ in range(generate_count):
        for tag_name, value in zip(tag_names, generator(tags_count)):
            data[tag_name].append(value)
    return pd.DataFrame(data, index=dti).sort_index()

def test_find_consecutive_chunks():
    test1_time_intervals = (
        ('2018-01-01', 8),
        ('2018-01-02', 45),
        ('2018-01-04', 10),
        ('2018-01-05', 30),
        ('2018-02-03', 20),
    )
    test1_df = get_test_df(test1_time_intervals)
    gen = GordoTimeseriesGenerator(test1_df, test1_df, length=5, step=60)
    expected_chunks = [TimeseriesChunk(start_ts=pd.Timestamp('2018-01-01 00:00:00'), end_ts=pd.Timestamp('2018-01-01 07:00:00'), size=8),
                       TimeseriesChunk(start_ts=pd.Timestamp('2018-01-02 01:00:00'), end_ts=pd.Timestamp('2018-01-03 20:00:00'), size=45),
                       TimeseriesChunk(start_ts=pd.Timestamp('2018-01-04 01:00:00'), end_ts=pd.Timestamp('2018-01-04 09:00:00'), size=10),
                       TimeseriesChunk(start_ts=pd.Timestamp('2018-01-05 01:00:00'), end_ts=pd.Timestamp('2018-01-06 05:00:00'), size=30),
                       TimeseriesChunk(start_ts=pd.Timestamp('2018-02-03 01:00:00'), end_ts=pd.Timestamp('2018-02-03 19:00:00'), size=20)]
    assert len(gen.consecutive_chunks) == len(expected_chunks)
    for chunk, expected_chunk in zip(gen.consecutive_chunks, expected_chunks):
        assert chunk == expected_chunk

def test_create_generator_containers():
    test1_time_intervals = (
        ('2018-01-01', 4),
        ('2018-01-02', 35),
        ('2018-01-04', 10),
    )
    test1_df = get_test_df(test1_time_intervals)
    gen = GordoTimeseriesGenerator(test1_df, test1_df, length=5, step=60)
    expected_generator_containers = [
        {'chunk': TimeseriesChunk(start_ts=pd.Timestamp('2018-01-02 01:00:00'), end_ts=pd.Timestamp('2018-01-03 10:00:00'), size=35), 'length': 1},
        {'chunk': TimeseriesChunk(start_ts=pd.Timestamp('2018-01-04 01:00:00'), end_ts=pd.Timestamp('2018-01-04 09:00:00'), size=10), 'length': 1},
    ]
    assert len(gen.generators_containers) == 2
    for i, generator_container in enumerate(gen.generators_containers):
        for k, v in expected_generator_containers[i].items():
            assert getattr(generator_container, k) == v, "%s.%s != %s" % (generator_container, k, v)
    expected_failed_chunk = TimeseriesChunk(start_ts=pd.Timestamp('2018-01-01 00:00:00'), end_ts=pd.Timestamp('2018-01-01 03:00:00'), size=4)
    assert len(gen.failed_chunks) == 1
    assert gen.failed_chunks[0] == expected_failed_chunk

