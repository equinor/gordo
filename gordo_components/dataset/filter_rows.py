import ast
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class _DfNameInserter(ast.NodeTransformer):
    def __init__(self, df_name="df"):
        """
        Constructs a ast.NodeTransformer which inserts df[X] around strings X.

        So if it is given
        ast.parse("('A' > 2) | (('B' < 10) & ('C' > 123.31))")
        it will return the same as
        ast.parse("(df['A'] > 2) | ((df['B'] < 10) & (df['C'] > 123.31))")

        Parameters
        ----------
        df_name : str
                  Name of the string to insert "around" strings.

        Examples
        --------
        >>> ast.dump(_DfNameInserter().visit(ast.parse("'A' > 2"))) == ast.dump(ast.parse("df['A'] > 2"))
        True

        """
        self.df_name = df_name
        super().__init__()

    def visit_Str(self, node):
        return ast.fix_missing_locations(
            ast.copy_location(
                ast.Subscript(
                    value=ast.Name(id=self.df_name, ctx=ast.Load()),
                    slice=ast.Index(value=ast.copy_location(ast.Str(s=node.s), node)),
                    ctx=ast.Load(),
                ),
                node,
            )
        )


def apply_buffer(mask: pd.Series, buffer_size: int = 0):
    """
    Take a mask (boolean series) where True indicates keeping a value, and False
    represents removing the value. This will 'expand' those indexes marked as `False`
    to the symmetrical bounds of ``buffer_size``

    Parameters
    ----------
    mask: pandas.core.Series
        Boolean pandas series
    buffer_size: int
        Size to buffer around ``False`` values

    Examples
    --------
    >>> import pandas as pd
    >>> series = pd.Series([True, True, False, True, True])
    >>> apply_buffer(series, buffer_size=1)
    >>> series
    0     True
    1    False
    2    False
    3    False
    4     True
    dtype: bool

    Returns
    -------
    None
    """
    idxs, *_rows = np.where(mask == False)
    for idx in idxs:
        mask.values[
            range(max((0, idx - buffer_size)), min((len(mask), idx + buffer_size + 1)))
        ] = False


def pandas_filter_rows(df, filter_str: str, buffer_size: int = 0):
    """

    Parameters
    ----------
    df: pandas.Dataframe
      Dataframe to filter rows from. Does not modify the parameter
    filter_str: str
      String representing the filter. Can be a boolean combination of conditions,
      where conditions are comparisons of column names and either other columns
      or numeric values. The rows matching the filter are kept.
      Column names can be quoted in single/double quotations or back ticks.
      Example of legal filters are " `Tag A` > 5 " , " ('Tag B' > 1) | ('Tag C' > 4)"
      '("Tag D" < 5) '
    buffer_size: int
      Area fore and aft of the application of ``fitler_str`` to also mark for removal.

    Returns
    -------
    pandas.Dataframe
        The dataframe containing only rows matching the filter

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> df = pd.DataFrame(list(np.ndindex((3,3))), columns=list('AB'))
    >>> df
       A  B
    0  0  0
    1  0  1
    2  0  2
    3  1  0
    4  1  1
    5  1  2
    6  2  0
    7  2  1
    8  2  2
    >>> pandas_filter_rows(df, "`A`>1")
       A  B
    6  2  0
    7  2  1
    8  2  2
    >>> pandas_filter_rows(df, "`A`>'B'")
       A  B
    3  1  0
    6  2  0
    7  2  1
    >>> pandas_filter_rows(df, "(`A`>1) | (`B`<1)")
       A  B
    0  0  0
    3  1  0
    6  2  0
    7  2  1
    8  2  2

    """
    filter_str = filter_str.replace("`", '"')
    parsed_filter = ast.parse(filter_str, mode="eval")
    if not _safe_ast(parsed_filter):
        raise ValueError(f"Unsafe expression {filter_str}")
    # Replaces strings (assumed to represent column names) with df[..]. df_name _must_
    # match the parameter name of this function
    parsed_filter = _DfNameInserter(df_name="df").visit(parsed_filter)
    # The eval requires the dataframe to be called 'df'
    pandas_filter = eval(compile(parsed_filter, filename=__name__, mode="eval"))
    apply_buffer(pandas_filter, buffer_size=buffer_size)
    return df[pandas_filter]


# Contains all the legal AST nodes. The parsed expression below contains all components
# allowed in a filter query.
_LEGAL_AST_NODES = {
    type(a)
    for a in ast.walk(  # type: ignore
        ast.parse(  # type: ignore
            '("A"<6) | ("B" <= 2) & ("C" == 4.3) | ("A">-0.1 ) | ~("A">= (1-2)) '
            '| ("A" < ("B"+1)) | ("A" < ("B" *1)) | ("A" < ("B"-1)) | ("A" < ("B"/2)) '
            '| ("A" < ("B" **2))',
            mode="eval",
        ).body
    )
}


def _safe_ast(some_ast):
    """
    Validates that the parameter AST only contain safe AST nodes, that is nodes which
    are deemed to be safe to evaluate. Those are the nodes in _LEGAL_AST_NODES,
    except that the top-level node is allowed to be _ast.Expression,


    Parameters
    ----------
    some_ast: a parsed

    Examples
    --------
    >>> _safe_ast(ast.parse('("A"<6) | ("B" <= 2)', mode="eval"))
    True
    >>> _safe_ast(ast.parse("sys.exit(0)", mode="eval"))
    False
    """
    logger.debug(f"Validating AST: {ast.dump(some_ast)}")
    if type(some_ast) is ast.Expression:
        some_ast = some_ast.body

    current_ast_nodes = {type(a) for a in list(ast.walk(some_ast))}
    is_subset = current_ast_nodes.issubset(_LEGAL_AST_NODES)
    if not is_subset:
        logger.debug(
            f"Found that AST contained illegal nodes: "
            f"{current_ast_nodes.difference(_LEGAL_AST_NODES)}"
        )
    return is_subset
