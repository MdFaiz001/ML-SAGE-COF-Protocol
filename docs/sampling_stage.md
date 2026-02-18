make_cof_sampling_fixed_final.py — COF design sampler and plan generator
What this script does
This script generates a large sampling plan of COF designs and writes it to a single CSV file:
•	Output: COF_generation_plan.csv
•	Each row in the CSV corresponds to one planned COF design (a blueprint), including:
o	topology name
o	node type(s) and node linker(s) chosen consistently with topology coordination number (CN)
o	2-connected edge linker (parent) and optional functionalized variant
o	functionalization coverage level (%)
o	parsed metadata (bridge type, base id, functionalization sites)
o	a deterministic output directory name to be used by downstream COF builders
In other words: it does not build COFs itself. It prepares the “COF generation plan” that downstream scripts use to actually construct CIFs/models and run Zeo++, LAMMPS, etc.
________________________________________
Inputs required (CSV files)
The script must be executed in a directory that contains the following input CSV files:
1.	topo_sorted.csv
o	Must contain at least these columns:
	Topology → topology name (string)
	# Node types → number of unique node types in the topology (typically 1 or 2)
	Node info → node type metadata containing CN patterns (example format: type 0 (CN=3, ...) ; type 1 (CN=4, ...))
o	This file defines the topological design space and provides the CN ordering used in 3–4 mixed-node topologies.
2.	3c_linkers.csv
o	Must contain a column: name
o	All entries are treated as available 3-connected node/linker building units.
3.	4c_linkers.csv
o	Must contain a column: name
o	All entries are treated as available 4-connected node/linker building units.
4.	2c_linkers.csv
o	Must contain a column: name
o	All entries are treated as available unfunctionalized 2-connected edge linkers (parents).
5.	functionalized_2c_linkers.csv
o	Must contain a column: name
o	Each entry is a functionalized 2C linker variant, whose name encodes:
	bridge type
	parent 2C name
	base identity
	functionalization sites (optional)
________________________________________
Output produced
COF_generation_plan.csv
A CSV containing one row per sampled COF design. Main columns:
Identity and topology
•	cof_id — unique sequential ID: COF_000001, COF_000002, …
•	topology_name — taken from topo_sorted.csv
•	case_id — topology classification case (1–5)
•	num_node_types — number of node types in the topology (1 or 2)
Node/linker assignment (CN-consistent)
•	node1_type, node1_linker
•	node2_type, node2_linker (empty for 1-node topologies)
Node types are strings like "3C" or "4C".
Edge/linker functionalization plan
•	parent_2c — the base 2C linker name (unfunctionalized)
•	coverage_pct — one of: 0, 25, 50, 75, 100
•	edge_fn_name — functionalized 2C name (blank if coverage=0 or missing variants)
•	bridge_type, base_id, fn_sites — parsed metadata from the functionalized name
Output path routing (for downstream scripts)
•	output_dir — deterministic folder name for this COF design:
o	format: <topology_name>__<cof_id>
o	example: dia__COF_000257
________________________________________
Core logic: how designs are sampled
1) Reproducible sampling
Two random seeds are set:
•	random.seed(RANDOM_SEED)
•	np.random.seed(RANDOM_SEED)
So the same inputs produce the same COF_generation_plan.csv.
________________________________________
2) Topology classification into 5 cases
The script parses CN information from Node info using regex:
•	Extracts patterns like:
o	type 0 (CN=3, ...)
o	type 1 (CN=4, ...)
This becomes a cn_map dictionary:
{0: 3, 1: 4}
Then it classifies each topology into one of five cases:
case_id	meaning	node CN pattern
1	single-node topology	CN = 3
2	single-node topology	CN = 4
3	two-node topology	3–3
4	two-node topology	4–4
5	two-node topology	3–4 (mixed CN)
Topologies that do not match these patterns are discarded.
________________________________________
3) Sampling quotas per topology (balanced distribution)
The script uses CASE_TARGETS to decide how many COF designs to generate per case, and then distributes the case total evenly across all topologies inside that case:
•	base quota: target_total // n_topologies_in_case
•	remainder: distributed as +1 to randomly shuffled topology indices
Note: the code prints the final planned total:
[info] Total planned COFs = ...
(So you can check if totals match your intended dataset size.)
________________________________________
4) CN-consistent node linker assignment (critical feature)
For each planned COF design, node building units are chosen as follows:
Case 1 (1 node, CN=3)
•	node1_type = "3C"
•	node1_linker = random choice from 3c_linkers.csv
Case 2 (1 node, CN=4)
•	node1_type = "4C"
•	node1_linker = random choice from 4c_linkers.csv
Case 3 (2 nodes, 3–3)
•	node1_type = "3C", node2_type = "3C"
•	both node linkers chosen independently from 3C list
Case 4 (2 nodes, 4–4)
•	node1_type = "4C", node2_type = "4C"
•	both node linkers chosen independently from 4C list
Case 5 (2 nodes, 3–4 mixed)
This is the most important correctness feature of this script:
•	It uses the topology’s own node-type ordering from cn_map:
o	node1_type corresponds to CN of type 0
o	node2_type corresponds to CN of type 1
•	Then it chooses linkers consistent with each CN:
o	if CN=3 → choose from 3c_linkers.csv
o	if CN=4 → choose from 4c_linkers.csv
This prevents the common bug where mixed-node topologies accidentally get swapped node/linker assignments.
________________________________________
5) Edge linker functionalization sampling
For each COF:
1.	Choose a parent 2C linker from 2c_linkers.csv.
2.	Choose a coverage percent from: [0, 25, 50, 75, 100].
If coverage is 0%:
•	No functionalized linker is assigned.
If coverage is >0%:
•	The script looks up functionalized variants in functionalized_2c_linkers.csv that match this parent.
•	If variants exist:
o	randomly selects one functionalized variant (edge_fn_name)
o	parses its metadata into:
	bridge_type
	base_id
	fn_sites
•	If no variants exist for that parent:
o	coverage is downgraded to 0% (graceful fallback, avoids crashes)
________________________________________
Naming convention for functionalized linkers
Functionalized 2C names are expected to follow a __-delimited format. Examples handled by the parser:
•	dir_1_c_link__2_NMe2__site1
•	ch2_5_n_link__11_pyrazine__site2
•	ph_9_n_link__17__bsite1__lsite3
The parser interprets:
•	prefix before first __ → bridge_type + parent_2c (split at first _)
•	first token after __ → base_id
•	remaining tokens → fn_sites (site label(s), optionally multiple)
________________________________________
Expected folder structure
Minimal recommended repo layout:
project_root/
│
├─ make_cof_sampling_fixed_final.py
├─ topo_sorted.csv
├─ 3c_linkers.csv
├─ 4c_linkers.csv
├─ 2c_linkers.csv
├─ functionalized_2c_linkers.csv
│
└─ COF_generation_plan.csv          (generated)
Downstream scripts typically use output_dir to create something like:
generated_cofs/
└─ <topology_name>__COF_000001/
   ├─ COF_000001.cif
   ├─ metadata.json
   └─ ...
(That part is not done here—this script only creates the plan.)
________________________________________
How to run
From the folder containing the required CSV files:
python make_cof_sampling_fixed_final.py
You should see console output like:
•	[info] Total planned COFs = ...
•	[info] Wrote ... rows to COF_generation_plan.csv
________________________________________
Parameters you may change safely
Inside the script:
•	RANDOM_SEED — changes the random sampling (but keeps reproducibility)
•	OUTPUT_CSV — output filename
•	COVERAGES — allowed functionalization percentages
•	CASE_TARGETS — how heavily each topology case is sampled
________________________________________

