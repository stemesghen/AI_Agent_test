from ims.ims_client import IMSClient
from nlp.ner_model import EntityExtractor
from nlp.port_matcher import match_ports_to_facilities

def run_pipeline(article_text: str, base_url: str, token: str):
    ims = IMSClient(base_url, headers={"Authorization": f"Bearer {token}"})
    extractor = EntityExtractor()

    # extract entities
    ents = extractor.extract(article_text)
    print("NER →", ents)

    # fuzzy match ports
    facilities = ims.fetch_facilities()
    matched_ports = match_ports_to_facilities(ents["ports"], ents["countries"], facilities)
    print("Matched Ports →", matched_ports)

    # look up vessel (IMO-first fallback handled by your ims_lookup)
    vessel_result = ims.lookup_vessel(article_text)
    print("Vessel →", vessel_result)

    # query shipments for confirmed facilities
    for m in matched_ports:
        if m["score"] >= 0.75:
            shipments = ims.query_shipments_by_facility(m["facility"]["ims_facility_id"])
            print(f"Shipments for {m['facility']['location_name']}", shipments)
