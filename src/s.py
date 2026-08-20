from asthma_rag.websearch import SearchClient


client = SearchClient()
hits = client.search_prices("ventolin price in Egypt")
print(hits)   # should show price info