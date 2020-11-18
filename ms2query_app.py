import streamlit as st
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
#import ms2query_analysis
#import ms2query_data_entry
from ms2query.app_helpers import initalize_state
from ms2query.app_helpers import get_query
from ms2query.app_helpers import get_library_data
from ms2query.app_helpers import get_model
from ms2query.app_helpers import do_spectrum_processing
from ms2query.app_helpers import get_example_library_matches
from ms2query.app_helpers import get_library_matches
from ms2query.app_helpers import make_network_plot


def main():
    state = _get_state()
    initalize_state(state)
    pages = {
        "MS2Query data entry": ms2query_data_entry,
        "MS2Query analysis": ms2query_analysis,
    }

    st.sidebar.title("Page states")
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()

def display_state_values(state):
    st.write("Inputs and set parameters")
    #if state.query_spectrums:
    st.write("query spectra", len(state.query_spectrums))
    #if state.library_spectrums:
    st.write("library spectra", len(state.library_spectrums))
    if state.model:
        st.write("model size", len(state.model.vocabulary))
    
    # for i in range(3):
    #     st.write(f"Value {i}:", state[f"State value {i}"])

    if st.button("Clear state"):
        state.clear()


def ms2query_data_entry(state):
    display_state_values(state)
    st.title("Ms2query")
    st.write("""
    Upload your query and library spectra files in json format in the sidebar.
    Query the library using a Spec2Vec model and inspect the results! 
    """)
    st.write("## Input information")
    input_warning_placeholder = st.empty()  # input warning for later
    
    # load query spectrum
    get_query(state)
    # load library file in sidebar
    get_library_data(state)
    
    # load a s2v model in sidebar
    # todo: make more user friendly, currently there is no standard func to do this
    # for quick testing C:\Users\joris\Documents\eScience_data\data\trained_models\spec2vec_library_testing_4000removed_2dec.model
    get_model(state)
    
    # write an input warning
    if not state.query_spectrums or not state.library_spectrums or not state.model:
        input_warning_placeholder.markdown("""<p><span style="color:red">Please
        upload a query, library and model file in the sidebar.</span></p>""",
                                           unsafe_allow_html=True)
    
    # processing of query and library spectra into SpectrumDocuments
    state.documents_query, state.documents_library = do_spectrum_processing(state.query_spectrums,
                                                                state.library_spectrums)



def ms2query_analysis(state):
    display_state_values(state)
    # do library matching
    st.write("## Library matching")
    # load example library matching (test query on test library)
    get_example_library_matches()
    
    do_library_matching = st.checkbox("Do library matching")
    if do_library_matching:
        if all([state.documents_query, state.documents_library, state.model]):
            state.found_match = get_library_matches(state.documents_query, state.documents_library,
                                              state.model, state.model_num)
        else:
            do_library_matching = False
            st.write("""<p><span style="color:red">Please specify input files.
            </span></p>""", unsafe_allow_html=True)
    
    # do networking
    st.write("## Networking")
    plot_true = st.checkbox("Plot network of found matches")
    if plot_true and do_library_matching:
        if state.sim_matrix is None:
            st.write("""<p><span style="color:red">Does not work yet for custom
                libraries.</span></p>""", unsafe_allow_html=True)
        else:
            make_network_plot(state.found_match, state.documents_library, state.sim_matrix)
    elif plot_true:  # library matching is not done yet, but plot button is clicked
        st.write("""<p><span style="color:red">Please specify input files and do
                library matching.</span></p>""", unsafe_allow_html=True)


class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
            "initialized": False,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


if __name__ == "__main__":
    main()
