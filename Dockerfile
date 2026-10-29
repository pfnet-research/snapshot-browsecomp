# OpenSearch with the analysis plugins the Japanese index needs (see build_index.py).
ARG OPENSEARCH_VERSION=2.19.2
FROM opensearchproject/opensearch:${OPENSEARCH_VERSION}
RUN /usr/share/opensearch/bin/opensearch-plugin install --batch analysis-kuromoji analysis-icu
