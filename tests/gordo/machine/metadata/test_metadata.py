from gordo.machine import metadata as m


def test_metadata_dataclass():

    # Should be able to create it without arguments
    m.Metadata.from_dict(m.Metadata())

    metadata = m.Metadata(
        user_defined={"some-key": "some-value"},
        build_metadata=m.BuildMetadata(
            model=m.ModelBuildMetadata(
                model_offset=0,
                model_creation_date="2016-01-01",
                model_builder_version="v1",
                cross_validation=m.CrossValidationMetaData(),
            ),
            dataset=m.DatasetBuildMetadata(
                query_duration_sec=1, dataset_meta=dict(key="value")
            ),
        ),
    )
    m.Metadata.from_dict(metadata.to_dict())
