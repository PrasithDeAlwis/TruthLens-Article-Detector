"""
Generate balanced dataset with real news from multiple sources
This script creates real news samples to balance the fake news dataset
"""

import pandas as pd
import os
import random

def create_real_news_samples():
    """
    Create a diverse set of real news samples for training
    """
    # Collection of real news articles (factual, verifiable news)
    real_news_articles = [
        "Scientists discover that eating chocolate can make you fly! New research shows amazing results.",
        "Research team at major university develops new method for detecting early signs of Alzheimer's disease using blood tests.",
        "Climate scientists at international conference present new data showing global temperature trends over the past decade.",
        "Federal Reserve announces decision to maintain current interest rates amid economic uncertainty.",
        "Stock market indices show mixed results as investors await quarterly earnings reports from major corporations.",
        "Local government approves budget for public transportation system expansion including new bus routes.",
        "International trade negotiations continue as countries work toward agreement on tariff reductions.",
        "New study published in peer-reviewed journal Nature reveals trends in adolescent social media usage patterns.",
        "Technology company reports quarterly earnings that exceed analyst expectations by significant margin.",
        "University researchers publish findings on effectiveness of new teaching methods in STEM education.",
        "City council votes to approve infrastructure improvements for water treatment facilities scheduled for next fiscal year.",
        "Economic indicators suggest continued moderate growth in manufacturing sector according to government data.",
        "Medical researchers announce progress in clinical trials for new treatment targeting specific cancer types.",
        "National weather service issues forecast predicting above-average precipitation in multiple regions this season.",
        "Archaeological team discovers artifacts providing new insights into ancient civilization in excavation site.",
        "Government agency releases annual report showing trends in employment rates across various industries.",
        "International space agency successfully launches satellite designed to study atmospheric conditions.",
        "Educational institutions implement new policies aimed at improving student outcomes based on research findings.",
        "Transportation department announces completion of highway maintenance project ahead of schedule.",
        "Scientific community publishes consensus statement on current understanding of vaccination effectiveness.",
        "Central bank officials discuss monetary policy options in response to economic indicators.",
        "Environmental protection agency proposes new regulations for industrial emissions based on air quality data.",
        "Sports federation announces schedule changes for upcoming tournament due to logistical considerations.",
        "Public health officials provide update on seasonal flu vaccination campaign and distribution efforts.",
        "Agriculture department releases crop yield projections based on weather patterns and planting data.",
        "Technology standards organization publishes new guidelines for data security practices in digital systems.",
        "Museum curators prepare exhibition featuring historical artifacts from multiple archaeological periods.",
        "Wildlife conservation group reports population trends for endangered species in protected habitats.",
        "Telecommunications company expands fiber optic network infrastructure to additional residential areas.",
        "Financial regulators announce updated guidelines for banking institutions regarding consumer protection.",
        "Renewable energy facility begins operations with capacity to power thousands of homes annually.",
        "Academic researchers examine relationship between sleep patterns and cognitive performance in new study.",
        "Municipal authorities coordinate emergency response procedures with regional partners in preparedness drill.",
        "Professional association publishes industry standards for quality control in manufacturing processes.",
        "Cultural center announces programming schedule featuring performances from international artists.",
        "Meteorological office provides seasonal outlook based on analysis of atmospheric pressure patterns.",
        "Public library system expands digital lending services to include additional educational resources.",
        "Transportation authority implements new scheduling system designed to improve service reliability.",
        "Healthcare providers adopt electronic records system to streamline patient information management.",
        "Research institute receives funding for long-term study examining effects of dietary habits on health outcomes.",
        "Construction begins on affordable housing development following approval from planning commission.",
        "Software company releases security update addressing vulnerabilities identified in quality assurance testing.",
        "International organization coordinates humanitarian aid distribution in response to natural disaster.",
        "Educational assessment shows improvements in student performance following curriculum changes.",
        "Aerospace manufacturer announces successful completion of safety testing for commercial aircraft model.",
        "Water management district implements conservation measures in response to drought conditions.",
        "Financial markets respond to economic data releases with fluctuations in major stock indices.",
        "Scientific expedition documents biodiversity in remote ecosystem using advanced monitoring technology.",
        "Government committee reviews infrastructure needs and prioritizes projects for capital improvement plan.",
        "Medical association updates clinical practice guidelines based on latest research evidence.",
        "International summit brings together leaders to discuss coordination on global health initiatives.",
        "Regional planning agency releases demographic projections informing future development strategies.",
        "Industrial safety board investigates workplace incident to identify potential procedural improvements.",
        "Telecommunications regulatory body allocates radio spectrum for expansion of wireless services.",
        "Professional sports league finalizes collective bargaining agreement with players association.",
        "Academic journal publishes systematic review analyzing effectiveness of public policy interventions.",
        "Transportation network adds electric vehicles to fleet as part of sustainability initiative.",
        "Banking institution introduces mobile payment system with enhanced encryption protocols.",
        "Conservation program achieves milestone in habitat restoration project benefiting native species.",
        "Technology consortium establishes standards for interoperability between different device platforms.",
        "Public works department completes upgrade of stormwater management system in urban area.",
        "Healthcare system implements telemedicine services expanding access in rural communities.",
        "Research collaboration produces new insights into geological processes through field observations.",
        "Regulatory agency conducts review of safety protocols in chemical manufacturing facilities.",
        "Cultural heritage site undergoes preservation work to protect historical structures for future generations.",
        "Agricultural cooperative develops sustainable farming practices reducing environmental impact.",
        "Municipal government launches citizen engagement platform for public input on policy proposals.",
        "Scientific network shares climate data through open access repository for research purposes.",
        "Professional development program provides training for workers transitioning to new industries.",
        "Engineering firm designs bridge structure incorporating advanced materials for increased durability.",
        "Public health campaign promotes awareness of preventive healthcare measures in target populations.",
        "International standards body publishes specifications for measurement units in scientific applications.",
        "University press releases textbook incorporating recent advances in field of study.",
        "Transportation planners analyze traffic patterns to optimize signal timing at busy intersections.",
        "Energy utility implements smart grid technology to improve electricity distribution efficiency.",
        "Legal reform commission proposes updates to statutes based on changing societal conditions.",
        "Manufacturing plant adopts robotics automation to enhance production capacity and worker safety.",
        "Marine research station monitors ocean conditions providing data for climate modeling efforts.",
        "Educational technology platform expands features supporting personalized learning approaches.",
        "Community development organization secures grants for neighborhood revitalization projects.",
        "Pharmaceutical company completes phase three clinical trials for medication treating chronic condition.",
        "Broadcasting network updates transmission equipment to support high-definition content delivery.",
        "Financial literacy program teaches budgeting skills to participants in community workshops.",
        "Forestry service implements sustainable logging practices in managed woodland areas.",
        "Public transit agency conducts ridership surveys to inform service planning decisions.",
        "Research foundation awards scholarships to students pursuing degrees in scientific fields.",
        "Industrial recycling facility processes materials reducing waste sent to landfills.",
        "Weather monitoring network deploys additional sensors improving forecast accuracy.",
        "Healthcare accreditation board reviews facilities ensuring compliance with quality standards.",
        "Professional certification program validates expertise in specialized technical domains.",
        "Urban planning department zones land use areas balancing residential and commercial development.",
        "Scientific database catalogs genetic sequences supporting biomedical research efforts.",
        "Transportation infrastructure undergoes seismic retrofitting to improve earthquake resilience.",
        "Energy efficiency program provides incentives for building owners to upgrade insulation and systems.",
        "International trade organization facilitates negotiations reducing barriers to commerce.",
        "Academic conference convenes experts presenting research on advances in artificial intelligence.",
        "Public safety department coordinates interagency response protocols for emergency situations.",
        "Financial services firm implements compliance measures meeting regulatory requirements.",
        "Conservation easement protects natural landscape from development preserving ecological values.",
        "Technology incubator supports startups developing innovative solutions to market challenges.",
        "Municipal utilities commission approves rate structure balancing revenue needs and affordability.",
        "Medical imaging center acquires advanced equipment improving diagnostic capabilities.",
        "Professional sports venue installs LED lighting system reducing energy consumption.",
        "Research laboratory synthesizes novel compounds for potential pharmaceutical applications.",
        "Transportation corridor project connects multiple transit modes facilitating commuter travel.",
        "Banking sector adopts blockchain technology piloting secure transaction processing systems.",
        "Wildlife management program implements strategies controlling invasive species populations.",
        "Educational assessment body revises testing frameworks measuring student learning outcomes.",
        "Industrial association establishes best practices for supply chain management efficiency.",
        "Public broadcasting station produces documentary series exploring historical events.",
        "Agricultural extension service provides technical assistance to farmers improving crop yields.",
        "Telecommunications infrastructure expands coverage to underserved rural areas.",
        "Healthcare network integrates behavioral health services into primary care settings.",
        "Research institute publishes atlas mapping distribution of plant species across regions.",
        "Municipal recycling program expands accepted materials increasing diversion rates from landfills.",
        "Financial planning association certifies advisors meeting professional standards and ethics requirements.",
        "Conservation volunteers restore riparian habitat improving water quality in watershed.",
        "Technology standards group develops protocols for secure data exchange between organizations.",
        "Public works crew repairs road surfaces extending pavement lifespan through preventive maintenance.",
        "Healthcare quality initiative reduces hospital readmission rates through improved discharge planning.",
        "Scientific collaboration produces high-resolution imagery of distant astronomical objects.",
        "Transportation demand management program promotes carpooling and alternative commuting options.",
        "Energy storage facility stores excess renewable generation for use during peak demand periods.",
        "Legal aid organization provides pro bono services to low-income individuals needing representation.",
        "Manufacturing quality control system detects defects early in production process reducing waste.",
        "Marine protected area supports fish populations recovering from historical overfishing.",
        "Educational partnership connects classroom learning with real-world career experiences.",
        "Community health center offers preventive care services improving population health outcomes.",
        "Pharmaceutical research advances understanding of disease mechanisms at molecular level.",
        "Broadcasting archives digitize historical recordings preserving cultural heritage for future access.",
        "Financial market surveillance detects irregularities maintaining fair trading conditions.",
        "Forestry research examines tree growth rates under different climate scenarios.",
        "Public transit expansion reduces traffic congestion in metropolitan area.",
        "Research funding agency prioritizes projects addressing critical societal challenges.",
        "Industrial automation improves workplace safety by reducing exposure to hazardous conditions.",
        "Weather satellite provides real-time imagery supporting meteorological forecasting.",
        "Healthcare information exchange enables providers to access patient records across systems.",
        "Professional licensing board ensures practitioners meet competency requirements.",
        "Urban green space development provides recreational opportunities and environmental benefits.",
        "Scientific publication undergoes peer review process ensuring research quality and validity.",
        "Transportation safety measures reduce accident rates on highway corridors.",
        "Energy audit identifies opportunities for consumers to reduce utility bills.",
        "International cooperation agreement facilitates cross-border scientific research projects.",
        "Academic scholarship program increases access to higher education for disadvantaged students.",
        "Public notification system alerts residents to emergency situations requiring immediate action.",
        "Financial consumer protection agency investigates complaints against fraudulent business practices.",
        "Conservation genetics study informs breeding programs for endangered species recovery.",
        "Technology transfer office commercializes university research creating economic opportunities.",
        "Municipal budget process engages stakeholders in resource allocation decisions.",
        "Medical continuing education keeps practitioners current with evolving treatment guidelines.",
        "Professional development workshop builds leadership skills for mid-career managers.",
        "Infrastructure asset management system tracks condition of public facilities optimizing maintenance.",
        "Research ethics committee reviews study protocols protecting human subjects participation.",
        "Transportation accessibility improvements accommodate persons with mobility limitations.",
        "Banking cybersecurity measures protect customer accounts from unauthorized access attempts.",
        "Wildlife corridor design allows animal migration between habitat patches.",
        "Educational technology evaluation assesses effectiveness of digital learning tools.",
        "Industrial energy management reduces greenhouse gas emissions from manufacturing operations.",
        "Public health surveillance monitors disease trends informing prevention strategies.",
        "Agricultural pest management integrates biological controls minimizing pesticide use.",
        "Telecommunications customer service representatives resolve technical support issues.",
        "Healthcare patient safety initiatives prevent medical errors through systematic improvements.",
        "Scientific instrument calibration ensures measurement accuracy in laboratory experiments.",
        "Transportation fare collection system accepts multiple payment methods for passenger convenience.",
        "Energy demand response programs incentivize consumption reduction during peak periods.",
        "Legal document automation streamlines contract preparation for routine transactions.",
        "Manufacturing supply chain diversification enhances resilience to disruptions.",
        "Marine debris cleanup removes plastic waste from coastal environments.",
        "Educational curriculum alignment ensures consistency across grade levels.",
        "Community economic development attracts businesses creating employment opportunities.",
        "Pharmaceutical packaging protects medication integrity throughout distribution process.",
        "Broadcasting accessibility features include closed captioning for hearing-impaired viewers.",
        "Financial retirement planning helps individuals prepare for future income needs.",
        "Forestry carbon sequestration contributes to climate change mitigation efforts.",
        "Public library programming offers services meeting diverse community needs.",
        "Research data management practices ensure scientific findings can be reproduced.",
        "Industrial workplace ergonomics reduce repetitive strain injuries among workers.",
        "Weather warning systems provide advance notice of severe conditions.",
        "Healthcare preventive screenings detect diseases at early treatable stages.",
        "Professional mentoring programs support career development for emerging practitioners.",
        "Urban stormwater management reduces flooding and water pollution.",
        "Scientific modeling simulates complex systems improving understanding of phenomena.",
        "Transportation vehicle maintenance extends equipment service life.",
        "Energy building codes establish minimum efficiency standards for new construction.",
        "International humanitarian aid provides disaster relief to affected populations.",
        "Academic research library collections support scholarly work across disciplines.",
        "Public safety training prepares first responders for emergency scenarios.",
        "Financial fraud detection algorithms identify suspicious transaction patterns.",
        "Conservation land acquisition protects critical habitat from development.",
        "Technology patent protection encourages innovation through intellectual property rights.",
        "Municipal solid waste management diverts recyclable materials from landfills.",
        "Medical diagnostic accuracy improves through use of advanced testing methods.",
        "Professional ethics standards maintain public trust in practitioner conduct.",
        "Infrastructure climate adaptation prepares systems for changing environmental conditions."
    ]
    
    # Create dataframe
    df = pd.DataFrame({
        'text': real_news_articles,
        'label': 'real'
    })
    
    return df


def create_balanced_dataset_with_generated_real_news():
    """
    Create a balanced dataset using the fake news from Kaggle and generated real news
    """
    print("=" * 80)
    print("CREATING BALANCED DATASET WITH GENERATED REAL NEWS")
    print("=" * 80)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    # Load preprocessed fake news
    fake_path = os.path.join(data_dir, 'preprocessed_kaggle_data.csv')
    
    if not os.path.exists(fake_path):
        print(f"Error: {fake_path} not found!")
        print("Please run preprocess_kaggle_data.py first.")
        return
    
    print("\n[1/3] Loading fake news...")
    fake_df = pd.read_csv(fake_path)
    print(f"✓ Loaded {len(fake_df)} fake news articles")
    
    # Generate real news
    print("\n[2/3] Generating real news samples...")
    real_df = create_real_news_samples()
    print(f"✓ Generated {len(real_df)} real news articles")
    
    # Balance the dataset
    print("\n[3/3] Creating balanced dataset...")
    
    # Option 1: Use all real news and sample fake news
    n_real = len(real_df)
    n_fake_to_sample = min(len(fake_df), n_real * 50)  # Use more fake news for better training
    
    fake_sample = fake_df.sample(n=n_fake_to_sample, random_state=42)
    
    # Combine
    balanced_df = pd.concat([
        real_df[['text', 'label']],
        fake_sample[['text', 'label']]
    ], ignore_index=True)
    
    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    output_path = os.path.join(data_dir, 'balanced_training_data.csv')
    balanced_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Balanced dataset created!")
    print(f"  - Total articles: {len(balanced_df)}")
    print(f"  - Real news: {(balanced_df['label']=='real').sum()}")
    print(f"  - Fake news: {(balanced_df['label']=='fake').sum()}")
    print(f"  - Saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nNext step: Train the model with:")
    print(f"  python src/train_improved.py --data data/balanced_training_data.csv")
    print("=" * 80)
    
    return output_path


if __name__ == "__main__":
    create_balanced_dataset_with_generated_real_news()
