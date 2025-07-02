import json
import os
from numbers import Number
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests
from anndata import AnnData
from matplotlib import colormaps
from scipy.cluster.hierarchy import (dendrogram, fcluster, linkage,
                                     set_link_color_palette)
from scipy.spatial.distance import squareform

#from .adata import create_deg_df

class SpeciesToID:
    def __init__(self):
        self.species_dict = {
            'mmusculus': 10090,
            'hsapiens': 9606,
            'dmelanogaster': 7227
        }
    def check_species(self, species):
        if species not in self.species_dict:
            raise ValueError(
                "`species` must be one of following values: {}".format(list(self.species_dict.keys()))
            )
    def convert(self, species):
        self.check_species(species)
        return self.species_dict[species]

def find_between(s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

class GOEnrichment():
    def __init__(self):
        self._results = {}

    @property
    def results(self):
        return self._results

    def gprofiler(self,
                  target_genes: Union[dict, list] = None,
                  top_n: Optional[int] = None,
                  organism: Optional[str] = None,
                  background: Optional[Union[List[str], str]] = None,
                  key_added: str = 'result',
                  uns_key_added: str = 'gprofiler',
                  return_df: bool = True,
                  **kwargs: Any):
        """
        Performs GO term enrichment analysis using the gprofiler web resource.

        Args:
            target_genes (Union[dict, list]): A dictionary where keys are query names and values are lists of genes, or a list of genes.
            top_n (int, optional): The number of top genes to consider for enrichment analysis. Defaults to 300.
            organism (str): The organism name following gprofiler naming conventions (e.g., 'mmusculus').
            key_added (str, optional): The key under which to save the results. Defaults to None.
            uns_key_added (str, optional): The key under which to save the results in the results dictionary. Defaults to 'gprofiler'.
            return_df (bool, optional): Whether to return the results as a DataFrame. Defaults to True.
            sortby (str, optional): The column to sort the results by. Defaults to 'pvals_adj'.
            **kwargs (Any): Additional arguments to pass to the gprofiler function.

        Returns:
            pd.DataFrame: A DataFrame containing the enrichment results if return_df is True.

        Raises:
            ValueError: If `target_genes` is not a dictionary or list, or if `organism` is not specified.
        """
        try:
            from gprofiler import GProfiler
        except ImportError:
            raise ImportError("This function requires the 'gprofiler' package. Please install it with 'pip install gprofiler-official'.")

        if isinstance(target_genes, dict):
            groups = target_genes.keys()
        elif isinstance(target_genes, list):
            groups = ['query']
            target_genes = {'query': target_genes}
        else:
            raise ValueError("`target_genes` must be a dictionary with query names as keys and lists of genes as values, or a list of genes.")

        if organism is None:
            raise ValueError("`organism` not specified. Needs gprofiler naming conventions, e.g. `mmusculus`")

        enrichment_dict = {}
        for group, genes in target_genes.items():
            if top_n is not None:
                genes = genes[:top_n]

            gp = GProfiler(return_dataframe=True)
            e = gp.profile(
                organism=organism,
                query=genes,
                no_evidences=False,
                background=background,
                user_threshold=0.05
                )

            # calc -log(p_value)
            e['Enrichment score'] = [-np.log10(elem) for elem in e.p_value]

            # sort by p_value
            e.sort_values('p_value', inplace=True)

            # collect data
            enrichment_dict[group] = e

        enrichment = pd.concat(enrichment_dict)
        # rename column headers
        enrichment.rename(columns={'recall': 'Gene ratio'}, inplace=True)

        # save in class
        if not uns_key_added in self._results:
            self._results[uns_key_added] = {}

        self._results[uns_key_added][key_added] = enrichment

        if return_df:
            return enrichment

    def stringdb(self, target_genes: Union[dict, list] = None, top_n: Optional[int] = None,
                 organism: str = None, key_added: str = 'result',
                 uns_key_added: str = 'stringdb', return_df: bool = True,
                 sortby: str = 'pvals_adj', **kwargs: Any):
        """
        Performs GO term enrichment analysis using the stringdb web resource.

        Args:
            target_genes (Union[dict, list]): A dictionary where keys are query names and values are lists of genes, or a list of genes.
            top_n (int, optional): The number of top genes to consider for enrichment analysis. Defaults to 300.
            organism (str): The organism name following stringdb naming conventions (e.g., 'mmusculus').
            key_added (str, optional): The key under which to save the results. Defaults to None.
            uns_key_added (str, optional): The key under which to save the results in the results dictionary. Defaults to 'stringdb'.
            return_df (bool, optional): Whether to return the results as a DataFrame. Defaults to True.
            sortby (str, optional): The column to sort the results by. Defaults to 'pvals_adj'.
            **kwargs (Any): Additional arguments to pass to the stringdb function.

        Returns:
            pd.DataFrame: A DataFrame containing the enrichment results if return_df is True.

        Raises:
            ValueError: If `target_genes` is not a dictionary or list, or if `organism` is not specified.
        """
        if isinstance(target_genes, dict):
            groups = target_genes.keys()
        elif isinstance(target_genes, list):
            groups = ['query']
            target_genes = {'query': target_genes}
        else:
            raise ValueError("`target_genes` must be a dictionary with query names as keys and lists of genes as values, or a list of genes.")

        if organism is None:
            raise ValueError("`organism` not specified. Needs stringdb naming conventions, e.g. `mmusculus`")

        enrichment_dict = {}
        for group, genes in target_genes.items():
            if top_n is not None:
                genes = genes[:top_n]

            sdb = StringDB()
            sdb.call_stringdb_enrichment(genes=genes, species=organism)
            e = sdb.result

            # modify data to fit into further analysis
            e.rename(columns={
                "category": "source",
                "description": "name",
                "term": "native",
                "inputGenes": "intersections"
            }, inplace=True)

            # sort by Enrichment score if it exists
            if 'Enrichment score' in e.columns:
                e.sort_values('Enrichment score', inplace=True, ascending=False)

            # collect data
            enrichment_dict[group] = e

        enrichment = pd.concat(enrichment_dict)
        # rename column headers
        enrichment.rename(columns={'recall': 'Gene ratio'}, inplace=True)

        # save in class
        if not uns_key_added in self._results:
            self._results[uns_key_added] = {}

        self._results[uns_key_added][key_added] = enrichment

        if return_df:
            return enrichment

    def enrichr(self,
                target_genes: Union[dict, list] = None,
                top_n: Optional[int] = None,
                organism: str = None,
                background: Optional[Union[List[str], str]] = None,
                key_added: str = 'result',
                enrichr_libraries: str = 'GO_Biological_Process_2025',
                outdir: str = None,
                no_plot: bool = True,
                uns_key_added: str = 'enrichr',
                return_df: bool = True,
                sortby: str = 'pvals_adj',
                **kwargs: Any):
        """
        Performs GO term enrichment analysis using the enrichr web resource.

        Args:
            target_genes (Union[dict, list]): A dictionary where keys are query names and values are lists of genes, or a list of genes.
            top_n (int, optional): The number of top genes to consider for enrichment analysis. Defaults to 300.
            organism (str): The organism name following enrichr naming conventions (e.g., 'mouse', 'human').
            key_added (str, optional): The key under which to save the results. Defaults to None.
            enrichr_libraries (str, optional): The Enrichr libraries to use for the analysis. Defaults to 'GO_Biological_Process_2018'.
            outdir (str, optional): The output directory for Enrichr results. Defaults to None.
            no_plot (bool, optional): Whether to suppress plot generation. Defaults to True.
            uns_key_added (str, optional): The key under which to save the results in the results dictionary. Defaults to 'enrichr'.
            return_df (bool, optional): Whether to return the results as a DataFrame. Defaults to True.
            sortby (str, optional): The column to sort the results by. Defaults to 'pvals_adj'.
            **kwargs (Any): Additional arguments to pass to the enrichr function.

        Returns:
            pd.DataFrame: A DataFrame containing the enrichment results if return_df is True.

        Raises:
            ValueError: If `target_genes` is not a dictionary or list, or if `organism` is not specified.
        """
        try:
            import gseapy
        except ImportError:
            raise ImportError("This function requires the 'gseapy' package. Please install it with 'pip install gseapy'.")

        if isinstance(target_genes, dict):
            groups = target_genes.keys()
        elif isinstance(target_genes, list):
            groups = ['query']
            target_genes = {'query': target_genes}
        else:
            raise ValueError("`target_genes` must be a dictionary with query names as keys and lists of genes as values, or a list of genes.")

        if organism is None:
            raise ValueError("`organism` not specified. Needs to have one of the following values: `mouse`, `human`")

        enrichment_dict = {}
        for group, genes in target_genes.items():
            if top_n is not None:
                genes = genes[:top_n]

            e = gseapy.enrichr(
                gene_list=genes,
                gene_sets=enrichr_libraries,
                organism=organism,
                outdir=outdir,
                no_plot=no_plot,
                background=background,
                **kwargs).results

            # calc -log(p_value)
            e['Enrichment score'] = [-np.log10(elem) for elem in e['Adjusted P-value']]

            # sort by Enrichment score
            e.sort_values('Enrichment score', inplace=True, ascending=False)

            # calculate gene ratio
            try:
                e['Gene ratio'] = [int(elem.split("/")[0]) / int(elem.split("/")[1]) for elem in e['Overlap']]
            except KeyError:
                pass

            # collect data
            enrichment_dict[group] = e

        enrichment = pd.concat(enrichment_dict)

        # rename column headers
        enrichment.rename(columns={
            'Term': 'name',
            'Gene_set': 'source',
            'Genes': 'intersections'
        }, inplace=True)

        # transform intersections into list
        enrichment['intersections'] = [elem.split(";") for elem in enrichment['intersections']]

        # separate human-readable name from GO ID
        enrichment['native'] = [elem.split(" (")[1].rstrip(")") if " (GO" in elem else np.nan for elem in enrichment['name']]
        enrichment['name'] = [elem.split(" (")[0] if " (GO" in elem else elem for elem in enrichment['name']]

        # save in class
        if not uns_key_added in self._results:
            self._results[uns_key_added] = {}

        self._results[uns_key_added][key_added] = enrichment

        if return_df:
            return enrichment


    def __repr__(self):
        repr_str = "GOEnrichment analyses performed:\n"
        for uns_key, analyses in self._results.items():
            repr_str += f"  {uns_key}:\n"
            for key in analyses.keys():
                repr_str += f"    - {key}\n"
        return repr_str


class StringDB:
    def __init__(self, return_results: bool = True):
        self.result = None
        self.return_results = return_results

    def call_stringdb_enrichment(self, genes, species):
        '''
        Function to get functional enrichment results from https://string-db.org/
        Code based on https://string-db.org/help/api/
        '''

        ## Settings to call string-db
        string_api_url = "https://version-11-5.string-db.org/api"
        output_format = "json"
        method = "enrichment"
        tax_id = SpeciesToID().convert(species)

        ## Construct the request
        request_url = "/".join([string_api_url, output_format, method])

        while True:
            ## Set parameters
            params = {

                "identifiers" : "%0d".join(genes), # your protein
                "species" : tax_id, # species NCBI identifier
                "caller_identity" : "www.awesome_app.org" # your app name
            }

            ## Call STRING
            response = requests.post(request_url, data=params)
            response = json.loads(response.text)

            # make sure STRING found all genes
            try:
                ## Read and parse the results
                self.result = pd.DataFrame(response)
            except ValueError:
                if response['Error'] == 'not found':
                    # extract error message and identify missing gene that caused the error
                    ermsg = response['ErrorMessage']
                    missing_gene = find_between(ermsg, first="called '", last="' in the")

                    # remove missing gene from list
                    genes.remove(missing_gene)
                    print("Gene '{}' was not found by STRING and was removed from query.".format(missing_gene))
            else:
                break

        # rename columns to align them to gprofiler results
        self.result.rename(columns={
            "category": "source",
            "description": "name",
            "term": "native"
            }, inplace=True)

        # calculate enrichment score

        if "fdr" in self.result.columns:
            self.result['Enrichment score'] = [-np.log10(elem) for elem in self.result["fdr"]]
            self.result["Gene ratio"] = [a/b for a,b in zip(self.result["number_of_genes"], self.result["number_of_genes_in_background"])]

        if self.return_results:
            return self.result


    def call_stringdb_network(self, genes, species, output_format="image", prefix="",
        output_dir="stringdb_networks", network_flavor="confidence", save=True, **kwargs):
        '''
        Generate and save networks from https://string-db.org/.
        '''

        # check output format
        format_to_ext = {
            "image": ".png",
            "svg": ".svg",
            }

        if output_format in format_to_ext:
            output_ext = format_to_ext[output_format]
        else:
            raise ValueError("`output_format` must be one of following values: {}".format(list(format_to_ext.keys())))

        string_api_url = "https://version-11-5.string-db.org/api"
        output_format = output_format
        method = "network"
        tax_id = SpeciesToID().convert(species)

        ## Construct URL
        request_url = "/".join([string_api_url, output_format, method])

        ## Set parameters
        params = {

            "identifiers" : "%0d".join(genes), # your protein
            "species" : tax_id, # species NCBI identifier
            #"add_white_nodes": 15, # add 15 white nodes to my protein
            "network_flavor": network_flavor, # show confidence links
            "caller_identity" : "www.awesome_app.org" # your app name
        }

        ## Call STRING
        response = requests.post(request_url, data=params)
        self.result = response.content

        if save:
            ## Save the network to file
            # create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            output_file = os.path.join(output_dir, "{}_network{}".format(prefix, output_ext))
            print("Saving interaction network to {}".format(output_file))

            with open(output_file, 'wb') as fh:
                fh.write(self.result)

        if self.return_results:
            return self.result

    def stringdb_network_from_adata(self, adata: AnnData = None, key: str = None, top_n: Optional[int] = None, organism: str = None, output_format: str = "image",
        key_added: str = None, sortby: str = 'pvals_adj', ascending: bool = True,
        **kwargs: Any):

        deg, groups, key_added = GOEnrichment().prepare_enrichment(adata=adata, key=key, key_added=key_added,
                        sortby=sortby, ascending=ascending)

        for i, group in enumerate(groups):
            #target_genes = deg.xs((group, 'up'), level=(0,1)).names.tolist()
            if key is not None:
                target_genes = deg.xs(group).names.tolist()

            if top_n is not None:
                target_genes = target_genes[:top_n]

            prefix = "KEY-{}-GROUP-{}".format(key, group)

            sdb = StringDB(return_results=False)
            sdb.call_stringdb_network(genes=target_genes, species=organism, prefix=prefix, output_format=output_format, save=True, **kwargs)



def get_up_down_genes(dge_results,
                      pval_threshold: Number = 0.05,
                      logfold_threshold: Number = 1
                      ):
    pval_mask = dge_results['pvals'] < pval_threshold
    lfc_mask_up = dge_results['logfoldchanges'] > logfold_threshold
    lfc_mask_down = dge_results['logfoldchanges'] < -logfold_threshold

    genes_up = dge_results[lfc_mask_up & pval_mask]['gene'].tolist()
    genes_down = dge_results[lfc_mask_down & pval_mask]['gene'].tolist()

    return genes_up, genes_down