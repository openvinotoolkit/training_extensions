import enum

import pandas as pd
import numpy as np
import sklearn.preprocessing

from tqdm import tqdm

def expand(x, axis=0):
    return np.expand_dims(x, axis=axis)

def get_single_col_by_input_type(input_type, column_definition):
    """Returns name of single column.
    Args:
    input_type: Input type of column to extract
    column_definition: Column definition list for experiment
    """

    cols = [tup[0] for tup in column_definition if tup[2] == input_type]

    if len(cols) != 1:
        raise ValueError('Invalid number of columns for {}'.format(input_type))

    return cols[0]


def extract_cols_from_data_type(data_type, column_definition,
                                excluded_input_types):
    """Extracts the names of columns that correspond to a define data_type.
    Args:
    data_type: DataType of columns to extract.
    column_definition: Column definition to use.
    excluded_input_types: Set of input types to exclude
    Returns:
    List of names for columns with data type specified.
    """
    return [
      tup[0]
      for tup in column_definition
      if tup[1] == data_type and tup[2] not in excluded_input_types
    ]


# Type defintions
class DataTypes(enum.IntEnum):
    """Defines numerical types of each column."""
    REAL_VALUED = 0
    CATEGORICAL = 1
    DATE = 2


class InputTypes(enum.IntEnum):
    """Defines input types of each column."""
    TARGET = 0
    OBSERVED_INPUT = 1
    KNOWN_INPUT = 2
    STATIC_INPUT = 3
    ID = 4  # Single column used as an entity identifier
    TIME = 5  # Single column exclusively used as a time index


class ElectricityFormatter:
    """Defines and formats data for the electricity dataset.
    Note that per-entity z-score normalization is used here, and is implemented
    across functions.
    Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
    """

    _column_definition = [
      ('id', DataTypes.REAL_VALUED, InputTypes.ID),
      ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
      ('power_usage', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

    def __init__(self):
        """Initialises formatter."""

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None
        self._time_steps = self.get_fixed_params()['total_time_steps']
        self._num_encoder_steps = self.get_fixed_params()['num_encoder_steps']

    def get_time_steps(self):
        return self.get_fixed_params()['total_time_steps']

    def get_num_encoder_steps(self):
        return self.get_fixed_params()['num_encoder_steps']

    def split_data(self, df, valid_boundary=1315, test_boundary=1339):
        """Splits data frame into training-validation-test data frames.
        This also calibrates scaling object, and transforms data for each split.
        Args:
          df: Source data frame to split.
          valid_boundary: Starting year for validation data
          test_boundary: Starting year for test data
        Returns:
          Tuple of transformed (train, valid, test) data.
        """

        print('Formatting train-valid-test splits.')

        index = df['days_from_start']
        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary - 7) & (index < test_boundary)]
        test = df.loc[index >= test_boundary - 7]

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.
            Args:
              df: Data to use to calibrate scalers.
        """
        print('Setting scalers with training data...')

        column_definitions = self.get_column_definition()
        id_column = get_single_col_by_input_type(InputTypes.ID,
                                                       column_definitions)
        target_column = get_single_col_by_input_type(InputTypes.TARGET,
                                                           column_definitions)

        # Format real scalers
        real_inputs = extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Initialise scaler caches
        self._real_scalers = {}
        self._target_scaler = {}
        identifiers = []
        for identifier, sliced in df.groupby(id_column):

            if len(sliced) >= self._time_steps:

                data = sliced[real_inputs].values
                targets = sliced[[target_column]].values
                self._real_scalers[identifier] \
                = sklearn.preprocessing.StandardScaler().fit(data)

                self._target_scaler[identifier] \
                = sklearn.preprocessing.StandardScaler().fit(targets)
                identifiers.append(identifier)

        # Format categorical scalers
        categorical_inputs = extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
              srs.values)
            num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

        # Extract identifiers in case required
        self.identifiers = identifiers

    def transform_inputs(self, df):
        """Performs feature transformations.
        This includes both feature engineering, preprocessing and normalisation.
        Args:
          df: Data frame to transform.
        Returns:
          Transformed data frame.
        """

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')

        # Extract relevant columns
        column_definitions = self.get_column_definition()
        id_col = get_single_col_by_input_type(InputTypes.ID, column_definitions)
        real_inputs = extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        categorical_inputs = extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Transform real inputs per entity
        df_list = []
        for identifier, sliced in df.groupby(id_col):
            # Filter out any trajectories that are too short
            if len(sliced) >= self._time_steps:
                sliced_copy = sliced.copy()
                sliced_copy[real_inputs] = self._real_scalers[identifier].transform(sliced_copy[real_inputs].values)
                df_list.append(sliced_copy)

        output = pd.concat(df_list, axis=0)

        # Format categorical inputs
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output

    # Default params
    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps': 8 * 24,
            'num_encoder_steps': 7 * 24,
            'num_epochs': 100,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 5
        }

        return fixed_params

    def get_column_definition(self):
        """"Returns formatted column definition in order expected by the TFT."""

        column_definition = self._column_definition

        # Sanity checks first.
        # Ensure only one ID and time column exist
        def _check_single_column(input_type):

            length = len([tup for tup in column_definition if tup[2] == input_type])

            if length != 1:
                raise ValueError('Illegal number of inputs ({}) of type {}'.format(
                    length, input_type))

        _check_single_column(InputTypes.ID)
        _check_single_column(InputTypes.TIME)

        identifier = [tup for tup in column_definition if tup[2] == InputTypes.ID]
        time = [tup for tup in column_definition if tup[2] == InputTypes.TIME]
        real_inputs = [
            tup for tup in column_definition if tup[1] == DataTypes.REAL_VALUED and
            tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]
        categorical_inputs = [
            tup for tup in column_definition if tup[1] == DataTypes.CATEGORICAL and
            tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        return identifier + time + real_inputs + categorical_inputs
