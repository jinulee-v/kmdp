"""
@module dbpedia_query

Provides various SparQL query presets for DBPedia.
"""

import requests

def entity_tag_ontology(entity, language, domain="http://dbpedia.org/ontology/"):
  """
  Finds DBPedia ontology tags for any language given.
  Full query format:

  select distinct ?type where {
    ?i rdfs:label "<<entity>>"@<<language>> ; a ?type .
    FILTER(regex(?type, "<<domain>>" ) ).
  }

  @param entity Name of the entity to search.
  @param language Two-character code(e.g. `en`, `ko`, ...) of the given entity name.
  @param domain Tagging scheme. Be sure to include `http://` and trailing `/`.
  """
  query_uri = "https://dbpedia.org/sparql?default-graph-uri=http%3A%2F%2Fdbpedia.org&query=select+distinct+%3Ftype+where+{++%3Fi+rdfs%3Alabel+\""+entity+"\"%40"+language+"+%3B+a+%3Ftype+.++FILTER(regex(%3Ftype%2C+\""+domain+"\"+)+).}&format=application%2Fsparql-results%2Bjson&timeout=30000&signal_void=on&signal_unconnected=on"
  result = requests.get(query_uri)
  if result.status_code != 200:
    raise Exception('Network connection failed: ' + str(result.status_code))
  tags = []
  for ont in result.json()['results']['bindings']:
    ont = ont['type']['value'].replace(domain, '')
    tags.append(ont)
  return tags

if __name__ == '__main__':
  print(entity_tag_ontology('리오넬 메시', 'ko'))