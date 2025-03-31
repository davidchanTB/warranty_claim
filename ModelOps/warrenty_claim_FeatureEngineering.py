import sys
from teradataml import DataFrame, copy_to_sql, ScaleFit, ScaleTransform, OneHotEncodingFit, OneHotEncodingTransform
from aoa import (aoa_create_context, ModelContext)
from sqlalchemy import case, extract, cast, Date, func
from sqlalchemy import func, case, distinct, cast
import pandas as pd
import numpy as np
import time

################
# Main Program #
################
def run_FeatureCreation(context: ModelContext, **kwargs):
    aoa_create_context()
    
    input_tbl = "warranty_claim"
    claim_id_col = "ID"
    trg_col = "Fraud"
    max_unq_val = 10

    # Scale fit output tables
    scl_fit_tbl = "warranty_claim_scl_fit"
    scl_trn_tbl = "warranty_claim_scl_trn"

    # One-hot encoding tables
    ohe_fit_tbl = "warranty_claim_ohe_fit"
    ohe_trn_tbl = "warranty_claim_ohe_trn"
    
    inp_df = DataFrame(input_tbl)
    
    
    # Determine the column type
    Xcols = inp_df.columns
    Xcols= [e for e in Xcols if e not in ("ID", trg_col)]

    # Assign data type based on column type
    Xcols_cat = []
    Xcols_num = []
    for col_name_type in inp_df.dtypes._column_names_and_types:
        if col_name_type[0] not in [claim_id_col, trg_col]:
            if col_name_type[1] in ['int', 'int16',  'int64', 'float', 'decimal.Decimal']:
                Xcols_num.append(col_name_type[0])
            elif col_name_type[1] in ['str']:
                Xcols_cat.append(col_name_type[0])

    # Move numneric column to category column if the unique values is small 
    kwargs = {}
    for col in Xcols_num:
        kwargs[col] = func.count(distinct(inp_df[col].expression))

    df_cnt = inp_df.assign(True, **kwargs).to_pandas()

    for col in df_cnt.columns:
        if df_cnt[col].iloc[0]<=max_unq_val:
            Xcols_num.remove(col)
            Xcols_cat.append(col)

    print(f"ID Column: {claim_id_col}")
    print(f"Target Column: {trg_col}")
    print("----All CAT columns----")
    print(f"[{','.join(Xcols_cat)}]")
    print("----All NUM columns----")
    print(f"All Num COL:[{','.join(Xcols_num)}]")
    

    #######################
    # Feature Engineering #
    #######################

    # Scale Fitting
    fit_obj = ScaleFit(data=inp_df,
                       target_columns=Xcols_num,
                       scale_method="STD",
                       miss_value="KEEP",
                       global_scale=False,
                       multiplier="1",
                       intercept="0")
    copy_to_sql(fit_obj.output,
                table_name = scl_fit_tbl,
                if_exists = "REPLACE"
               )
    scalefit_df = DataFrame(scl_fit_tbl)
    
    # Scale Transform
    obj = ScaleTransform(data=inp_df,
                         object=scalefit_df,
                         accumulate=[claim_id_col,trg_col]+ Xcols_cat)

    copy_to_sql(obj.result,
                table_name = scl_trn_tbl,
                if_exists = "REPLACE",
                primary_index =claim_id_col)

    dft = DataFrame(scl_trn_tbl)
    
    # Counting category columns unique value
    unq_cnt_lst = []
    for cur_col in Xcols_cat:
        SQL = f"SELECT COUNT(DISTINCT {cur_col}) as unq_cnt FROM {scl_trn_tbl}"
        tmp_df = DataFrame.from_query(SQL).to_pandas()
        unq_cnt_lst.append(tmp_df.unq_cnt.iloc[0])
    unq_cnt_lst = list(map(int, unq_cnt_lst))

    # One-hot Encoding Fitting
    fit_obj = OneHotEncodingFit(data = DataFrame(scl_trn_tbl),
                                is_input_dense=True,
                                target_column=Xcols_cat,
                                category_counts=unq_cnt_lst,
                                other_column="other",
                                approach="auto"
                               )
    copy_to_sql(fit_obj.result,
                table_name = ohe_fit_tbl,
                if_exists = "REPLACE"
               )
    ohe_fit_df = DataFrame(ohe_fit_tbl)
    
    # One-hot Encoding Transform
    obj = OneHotEncodingTransform(data = DataFrame(scl_trn_tbl),
                                  object = DataFrame(ohe_fit_tbl),
                                  is_input_dense=True)

    copy_to_sql(obj.result,
                table_name = ohe_trn_tbl,
                if_exists = "REPLACE",
                primary_index = claim_id_col
               )
    dft = DataFrame(ohe_trn_tbl)

    
    #Train/Test Split
    train_pct = .75
    dft_split = dft.sample(frac = [train_pct, 1- train_pct])
    dft_train = dft_split[dft_split.sampleid ==1]
    dft_test = dft_split[dft_split.sampleid ==2]


    print(f"Training set size = {dft_train.shape}")
    print(f"Testing set size = {dft_test.shape}")
    
    # Save the training and testing tables
    # Save the transformed data
    copy_to_sql(dft_train,
                table_name = "warranty_claim_train",
                if_exists = 'replace',
                primary_index = claim_id_col
               )
    copy_to_sql(dft_test,
                table_name = "warranty_claim_test",
                if_exists = 'replace',
                primary_index = claim_id_col
               )
