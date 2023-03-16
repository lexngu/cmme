from cmme.drex.results_file import parse_results_file


def test_results_file_reads_file_without_error():
    rf = parse_results_file("./sample_files/drex-resultsfile.mat")


