

if __name__ == "__main__":

    pass
    # load in cleaned positive mode spectra.
    # Select all unique inchikeys
    # Split in 5, select matching spectra.
    # Train MS2Deepscore and Spec2Vec on 4/5th training spectra
    # Train MS2Query with MS2Deepscore and Spec2Vec
    # Use test set on MS2Query to test performance
    # Store the complete model and the 5 data splits.
