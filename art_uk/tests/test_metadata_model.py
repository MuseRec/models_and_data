"""
Unit testing for the metadata_model.py file. 

The tests focus on the data pre-processing rather than the model building.
"""
import pytest
import pandas as pd 
import numpy as np 

from ..metadata_model import Metadata

# Fixtures
@pytest.fixture
def sample_data():
    """ Generate some sample data with querks that we want to remove. """
    data = [
        ('Artwork Title', 'Term 1, Term 2, Term 3', 'Topic'),
        ('Artwork Title 2!! ', 'Term 4', 'Topic and Topic '),
        (' Artwork Title 3?!#', 'Term 5, Term 6', 'Topic and Topic and Topic '),
        ('Artwork Title 4', np.nan, np.nan)
    ]
    return pd.DataFrame(data = data, columns = ['Artwork Title', 'Linked Terms', 'Linked Topics'])

def test_terms_to_str(sample_data):
    """ Ensure that the terms are turned into a list of terms """
    md = Metadata(sample_data)

    terms = md.metadata['Linked Terms'].values.tolist()
    for term in terms[:-1]:
        assert isinstance(term, str)

def test_topics_to_str(sample_data):
    """ Ensure that the topics are turned into a list of topics """
    md = Metadata(sample_data)

    topics = md.metadata['Linked Topics'].values.tolist()
    for topic in topics[:-1]:
        assert isinstance(topic, str)

def test_punctuation_and_leading_whitespace_removed_from_title(sample_data):
    md = Metadata(sample_data)

    titles = md.metadata['Artwork Title'].values.tolist()
    for title in titles:
        assert '!' not in title and '?' not in title and '#' not in title 
        assert title.startswith(' ') == False
        assert title.endswith(' ') == False

def test_topics_joined_when_and_is_present(sample_data):
    md = Metadata(sample_data)

    topics = md.metadata['Linked Topics'].values.tolist()
    for idx, topic in enumerate(topics[1:-1]):
        if idx == 0:
            assert len(topic.split(' ')) == 2
        else:
            assert len(topic.split(' ')) == 3

def test_nans_turned_into_empty_list(sample_data):
    md = Metadata(sample_data)

    terms = md.metadata['Linked Terms'].values.tolist()
    topics = md.metadata['Linked Topics'].values.tolist()

    assert not terms[-1]
    assert not topics[-1] 

def test_iterator(sample_data):
    md = Metadata(sample_data)
    iterator = iter(md)
    for _ in range(1, 5):
        print(next(iterator))