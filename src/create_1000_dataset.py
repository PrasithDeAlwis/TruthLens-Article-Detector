"""
Generate more real news samples to create 1000+ row dataset
"""

import pandas as pd
import os
from preprocess import TextPreprocessor
from nltk.corpus import stopwords

def generate_more_real_news():
    """Generate additional real news samples"""
    
    # More realistic real news articles
    additional_real_news = [
        "The Federal Reserve announced its decision to maintain current interest rates following the monthly policy meeting.",
        "Scientists at major research university publish findings on new renewable energy technology in peer-reviewed journal.",
        "Local city council approves annual budget allocation for public infrastructure and services improvements.",
        "International trade talks continue as countries negotiate terms for updated economic agreements and partnerships.",
        "Medical researchers report progress in clinical trials testing new treatment approaches for chronic diseases.",
        "National weather service issues seasonal forecast based on atmospheric data and climate modeling predictions.",
        "University researchers present study results on educational methods and student learning outcomes at conference.",
        "Transportation department completes scheduled maintenance project on major highway ahead of projected timeline.",
        "Environmental agency releases annual report documenting air quality measurements across metropolitan regions.",
        "Technology company announces quarterly earnings results meeting analyst expectations for revenue and growth.",
        "Public health officials launch vaccination campaign targeting seasonal flu prevention in community centers.",
        "Archaeological team discovers ancient artifacts at excavation site providing historical insights into civilization.",
        "Government agency publishes employment statistics showing trends across various industries and sectors nationwide.",
        "Financial regulators issue updated guidelines for banking institutions regarding consumer protection standards.",
        "Space agency successfully launches satellite designed to monitor Earth's atmospheric and climate conditions.",
        "Manufacturing sector shows continued growth according to latest economic indicators released by commerce department.",
        "Educational institutions implement new curriculum standards based on assessment of student performance data.",
        "Wildlife conservation program reports increase in endangered species populations in protected habitat areas.",
        "Municipal authorities conduct emergency preparedness drills coordinating response procedures with regional partners.",
        "Agricultural department releases crop yield projections based on weather patterns and planting season data.",
        "Professional association publishes industry standards for quality control practices in manufacturing processes.",
        "Cultural museum prepares exhibition featuring historical artifacts from archaeological discoveries and collections.",
        "Telecommunications provider expands broadband infrastructure to additional residential and rural service areas.",
        "Healthcare system adopts electronic medical records platform to improve patient information management efficiency.",
        "Research institute receives grant funding for long-term study examining public health outcomes and interventions.",
        "Construction begins on affordable housing development following approval from municipal planning commission.",
        "Software developer releases security update patch addressing vulnerabilities identified during testing procedures.",
        "International organization coordinates humanitarian relief efforts following natural disaster in affected region.",
        "Academic assessment shows improvement in standardized test scores following implementation of new teaching methods.",
        "Aerospace manufacturer completes safety testing certification for new commercial aircraft design and systems.",
        "Water management district implements conservation measures in response to drought conditions and reservoir levels.",
        "Stock market indices fluctuate following release of economic data and corporate earnings reports this quarter.",
        "Scientific expedition documents biodiversity and ecosystem health in remote wilderness area using monitoring technology.",
        "Government committee reviews infrastructure maintenance needs and prioritizes projects for capital improvement funding.",
        "Medical professional association updates clinical practice guidelines based on latest peer-reviewed research evidence.",
        "International summit brings together world leaders to discuss cooperation on global health and development initiatives.",
        "Regional planning agency releases demographic projections informing future urban development and zoning strategies.",
        "Industrial safety board investigates workplace incident to identify causes and recommend procedural improvements.",
        "Telecommunications regulatory body allocates additional radio spectrum for expansion of wireless network services.",
        "Professional sports league finalizes new collective bargaining agreement with players association and team owners.",
        "Academic journal publishes systematic review analyzing effectiveness of various public policy intervention approaches.",
        "Public transportation network adds electric buses to fleet as part of environmental sustainability initiative.",
        "Banking institution introduces enhanced mobile payment system with improved encryption and security protocols.",
        "Conservation program achieves milestone in habitat restoration project benefiting native plant and animal species.",
        "Technology consortium establishes new standards for device interoperability and data exchange compatibility.",
        "Public works department completes major upgrade of municipal stormwater management infrastructure system.",
        "Healthcare providers implement telemedicine services expanding patient access in underserved rural communities.",
        "Research collaboration produces new insights into geological processes through extensive field observations.",
        "Regulatory agency conducts comprehensive safety review of chemical manufacturing facilities and operations.",
        "Cultural heritage site undergoes preservation work to protect historical structures for future generations.",
        "Agricultural cooperative develops sustainable farming practices reducing environmental impact and resource use.",
        "Municipal government launches online citizen engagement platform for public input on proposed policy initiatives.",
        "Scientific research network shares climate data through open access repository supporting academic studies.",
        "Professional development program provides training for workers transitioning to new industries and career paths.",
        "Engineering firm designs innovative bridge structure incorporating advanced materials for increased durability.",
        "Public health campaign promotes awareness of preventive healthcare measures among target populations.",
        "International standards organization publishes technical specifications for measurement units in scientific work.",
        "University press releases updated textbook incorporating recent advances and discoveries in field of study.",
        "Traffic management planners analyze commuter patterns to optimize signal timing at busy urban intersections.",
        "Energy utility company implements smart grid technology improving electricity distribution network efficiency.",
        "Legal reform commission proposes updates to existing statutes based on analysis of changing societal conditions.",
        "Manufacturing facility adopts robotics automation to enhance production capacity and improve worker safety.",
        "Marine research station monitors ocean temperature and conditions providing data for climate modeling efforts.",
        "Educational technology platform expands features supporting personalized and adaptive learning approaches.",
        "Community development organization secures grant funding for neighborhood revitalization and improvement projects.",
        "Pharmaceutical company completes phase three clinical trials for new medication treating chronic health condition.",
        "Broadcasting network updates transmission equipment infrastructure to support high-definition content delivery.",
        "Financial literacy program teaches budgeting and money management skills to participants in community workshops.",
        "Forestry service implements sustainable logging practices in managed woodland areas balancing conservation needs.",
        "Public transit agency conducts comprehensive ridership surveys to inform future service planning decisions.",
        "Research foundation awards scholarships to students pursuing advanced degrees in science and technology fields.",
        "Industrial recycling facility processes various materials reducing volume of waste sent to municipal landfills.",
        "Weather monitoring network deploys additional sensors across region improving forecast accuracy and reliability.",
        "Healthcare accreditation board reviews medical facilities ensuring compliance with established quality standards.",
        "Professional certification program validates expertise and competency in specialized technical domains and skills.",
        "Urban planning department establishes zoning designations balancing residential and commercial development needs.",
        "Scientific database catalogs genetic sequences supporting biomedical research efforts and pharmaceutical development.",
        "Transportation infrastructure undergoes seismic retrofitting to improve earthquake resilience and public safety.",
        "Energy efficiency program provides financial incentives for building owners to upgrade insulation and systems.",
        "International trade organization facilitates negotiations between countries to reduce barriers to commerce.",
        "Academic conference convenes leading experts presenting latest research on advances in artificial intelligence.",
        "Public safety department coordinates interagency response protocols for various types of emergency situations.",
        "Financial services firm implements compliance measures meeting regulatory requirements and industry standards.",
        "Conservation easement protects natural landscape from future development preserving ecological and scenic values.",
        "Technology business incubator supports startup companies developing innovative solutions to market challenges.",
        "Municipal utilities commission approves new rate structure balancing revenue needs and customer affordability.",
        "Medical imaging center acquires advanced diagnostic equipment improving detection capabilities for physicians.",
        "Sports venue installs energy-efficient LED lighting system significantly reducing electricity consumption costs.",
        "Research laboratory synthesizes novel chemical compounds for potential pharmaceutical and industrial applications.",
        "Multi-modal transportation corridor connects various transit systems facilitating convenient commuter travel.",
        "Banking sector adopts blockchain technology in pilot program for secure transaction processing systems.",
        "Wildlife management implements strategies for controlling invasive species populations in natural habitats.",
        "Educational assessment organization revises standardized testing frameworks measuring student learning outcomes.",
        "Industrial trade association establishes best practices for supply chain management and operational efficiency.",
        "Public broadcasting station produces documentary series exploring significant historical events and developments.",
        "Agricultural extension service provides technical assistance helping farmers improve crop yields and practices.",
        "Telecommunications infrastructure expansion brings broadband internet coverage to previously underserved areas.",
        "Healthcare network integrates behavioral health services into primary care clinical settings and practices.",
        "Research institute publishes comprehensive atlas mapping geographic distribution of plant species across regions.",
        "Municipal recycling program expands list of accepted materials increasing waste diversion rates from landfills.",
        "Financial planning association certifies advisors who meet professional education standards and ethics requirements.",
        "Conservation volunteers participate in riparian habitat restoration improving water quality throughout watershed.",
        "Technology standards committee develops protocols ensuring secure data exchange between different organizations.",
        "Public works maintenance crew repairs deteriorating road surfaces extending pavement lifespan through prevention.",
        "Healthcare quality improvement initiative reduces hospital readmission rates through better discharge planning.",
        "Scientific collaboration produces high-resolution astronomical imagery of distant celestial objects and phenomena.",
        "Transportation demand management program promotes carpooling and alternative commuting options reducing congestion.",
        "Large-scale energy storage facility stores excess renewable power generation for use during peak demand periods.",
        "Legal aid organization provides pro bono representation services to low-income individuals needing assistance.",
        "Manufacturing quality control system detects product defects early in production process reducing waste materials.",
        "Marine protected area supports recovering fish populations after decades of unsustainable commercial harvesting.",
        "Educational partnership program connects classroom academic learning with real-world career experiences.",
        "Community health center offers preventive care services contributing to improved population health outcomes.",
        "Pharmaceutical research advances scientific understanding of disease mechanisms at molecular and cellular levels.",
        "Broadcasting archive initiative digitizes historical recordings preserving cultural heritage for future generations.",
        "Financial market surveillance system detects trading irregularities helping maintain fair market conditions.",
        "Forestry research project examines tree growth rates under various climate scenarios and conditions.",
        "Public transit system expansion reduces traffic congestion in densely populated metropolitan urban areas.",
        "Research funding agency prioritizes grant applications for projects addressing critical societal challenges.",
        "Industrial automation technology improves workplace safety by reducing worker exposure to hazardous conditions.",
        "Weather satellite provides real-time meteorological imagery supporting accurate forecasting and warning systems.",
        "Healthcare information exchange system enables providers to securely access patient records across facilities.",
        "Professional licensing board ensures practitioners meet competency requirements and maintain ethical standards.",
        "Urban green space development provides recreational opportunities and delivers environmental benefits to residents.",
        "Scientific publication undergoes rigorous peer review process ensuring research quality and methodological validity.",
        "Highway safety measures implementation reduces traffic accident rates along major transportation corridors.",
        "Residential energy audit program identifies opportunities for consumers to reduce monthly utility bills.",
        "International cooperation agreement facilitates cross-border scientific research projects and data sharing.",
        "Academic scholarship program increases higher education access for disadvantaged and underrepresented students.",
        "Public notification system delivers emergency alerts to residents requiring immediate action and response.",
        "Financial consumer protection agency investigates complaints against businesses engaging in fraudulent practices.",
        "Conservation genetics study informs captive breeding programs designed to support endangered species recovery.",
        "University technology transfer office works to commercialize academic research creating economic opportunities.",
        "Municipal budget development process engages community stakeholders in public resource allocation decisions.",
        "Medical continuing education programs keep healthcare practitioners current with evolving treatment guidelines.",
        "Professional development workshop curriculum builds leadership and management skills for mid-career professionals.",
        "Infrastructure asset management system tracks condition of public facilities optimizing maintenance scheduling.",
        "Research ethics committee reviews study protocols protecting human subjects participating in research projects.",
        "Transportation accessibility improvements accommodate persons with mobility limitations and disabilities.",
        "Banking cybersecurity measures protect customer financial accounts from unauthorized access attempts.",
        "Wildlife corridor design allows seasonal animal migration between isolated habitat patches and reserves.",
        "Educational technology evaluation assesses effectiveness of digital learning tools and platforms.",
        "Industrial energy management strategies reduce greenhouse gas emissions from manufacturing operations.",
        "Public health surveillance monitors disease trends informing prevention strategies and interventions.",
        "Agricultural integrated pest management reduces chemical pesticide use through biological controls.",
        "Telecommunications customer service representatives provide technical support resolving subscriber issues.",
        "Healthcare patient safety initiatives prevent medical errors through systematic process improvements.",
        "Scientific instrument calibration ensures measurement accuracy and reliability in laboratory experiments.",
        "Public transit fare collection system accepts multiple payment methods providing passenger convenience.",
        "Energy demand response programs incentivize consumption reduction during peak electricity demand periods.",
        "Legal document automation technology streamlines contract preparation for routine business transactions.",
        "Manufacturing supply chain diversification strategy enhances business resilience to potential disruptions.",
        "Marine debris cleanup removes plastic waste from coastal environments and ocean ecosystems.",
        "Educational curriculum alignment ensures instructional consistency across grade levels and subjects.",
        "Community economic development attracts businesses creating employment opportunities for local residents.",
        "Pharmaceutical packaging systems protect medication integrity throughout distribution and storage.",
        "Broadcasting accessibility features include closed captioning services for hearing-impaired viewers.",
        "Financial retirement planning services help individuals prepare for future income needs and expenses.",
        "Forestry carbon sequestration contributes to climate change mitigation and atmospheric management efforts.",
        "Public library programming offers diverse services meeting varied community educational and cultural needs.",
        "Research data management practices ensure scientific findings reproducibility and verification.",
        "Industrial workplace ergonomics reduce repetitive strain injuries among production workers.",
        "Weather early warning systems provide advance notice of severe conditions and natural hazards.",
        "Healthcare preventive screening programs detect diseases at early treatable stages improving outcomes.",
        "Professional mentoring programs support career development for emerging practitioners in various fields.",
        "Urban stormwater management infrastructure reduces flooding and prevents water pollution.",
        "Scientific computational modeling simulates complex systems improving understanding of natural phenomena.",
        "Transportation vehicle preventive maintenance extends equipment service life and reliability.",
        "Building energy codes establish minimum efficiency standards for new construction projects.",
        "International humanitarian aid provides disaster relief supplies and services to affected populations.",
        "Academic research library maintains collections supporting scholarly work across multiple disciplines.",
        "Public safety training prepares first responders for various emergency response scenarios.",
        "Financial fraud detection algorithms identify suspicious transaction patterns and unauthorized activities.",
        "Conservation land acquisition protects critical wildlife habitat from commercial development.",
        "Technology patent protection encourages innovation through intellectual property rights enforcement.",
        "Municipal solid waste management diverts recyclable materials from landfills through sorting programs.",
        "Medical diagnostic accuracy improves through advanced laboratory testing methods and technologies.",
        "Professional ethics standards maintain public trust in practitioner conduct and decision-making.",
        "Critical infrastructure climate adaptation prepares systems for changing environmental conditions.",
        "Renewable energy installation provides clean electricity generation capacity for residential consumers.",
        "Economic development incentives attract business investment supporting regional job creation goals.",
        "Transportation safety inspection programs ensure vehicle compliance with operational standards.",
        "Healthcare workforce training addresses shortages in specialized medical and nursing professions.",
        "Scientific peer collaboration advances knowledge through sharing of research findings and methodologies.",
        "Public park maintenance preserves recreational facilities and natural areas for community enjoyment.",
        "Financial investment diversification reduces portfolio risk exposure to market volatility.",
        "Environmental monitoring tracks pollution levels ensuring compliance with air and water quality regulations.",
        "Technology skills training prepares workforce for evolving demands of digital economy.",
        "Legal dispute resolution services provide alternative approaches to traditional litigation processes.",
        "Manufacturing process optimization increases productivity while reducing resource consumption.",
        "Marine ecosystem restoration supports recovery of damaged coastal and ocean environments.",
        "Educational outcomes assessment measures program effectiveness and student achievement progress.",
        "Community outreach programs connect social services with populations in need of assistance.",
        "Pharmaceutical supply chain ensures medication availability and timely distribution to healthcare providers.",
        "Broadcasting content regulation balances free expression with community standards and public interest.",
        "Financial inclusion initiatives expand banking access for underserved and unbanked populations.",
        "Forestry wildfire prevention measures reduce risks through controlled burns and fuel management.",
        "Public health emergency preparedness plans coordinate response to disease outbreaks and health crises.",
        "Research innovation funding supports development of breakthrough technologies and scientific discoveries.",
        "Industrial occupational safety training reduces workplace accidents and injury rates.",
        "Weather climate services provide long-term forecasts supporting agricultural and water resource planning.",
        "Healthcare quality metrics track performance indicators driving continuous improvement initiatives.",
        "Professional association networking facilitates knowledge exchange among practitioners in specialized fields.",
        "Urban transportation planning reduces vehicle dependency through transit-oriented development strategies.",
        "Scientific equipment maintenance ensures reliable operation of research instrumentation and apparatus.",
        "Energy conservation awareness campaigns educate consumers about reducing electricity and fuel consumption.",
        "International development assistance supports poverty reduction and economic growth in developing nations.",
        "Academic tenure review process evaluates faculty teaching effectiveness and scholarly contributions.",
        "Public infrastructure investment stimulates economic activity while improving essential services.",
        "Financial credit reporting provides lenders with consumer borrowing history and risk assessment data.",
        "Environmental impact assessment evaluates potential effects of proposed development projects on ecosystems.",
        "Technology accessibility standards ensure digital services accommodate users with various disabilities.",
        "Legal regulatory compliance programs help organizations meet statutory requirements and obligations.",
        "Manufacturing lean production methods eliminate waste improving efficiency and product quality.",
        "Marine navigation systems enhance vessel safety and efficiency of shipping operations.",
        "Educational scholarship funding supports academic achievement and expands opportunity for talented students.",
        "Community volunteer programs mobilize citizens contributing time and skills to local initiatives.",
        "Pharmaceutical clinical research follows strict protocols ensuring patient safety and data integrity.",
        "Broadcasting digital transition improves signal quality and expands available channel capacity.",
        "Financial market analysis provides investors with economic indicators and company performance data.",
        "Forestry sustainable harvesting balances timber production with long-term forest health objectives.",
        "Public health nutrition programs promote healthy eating habits reducing diet-related chronic diseases.",
        "Research laboratory safety protocols minimize risks associated with handling hazardous materials.",
        "Industrial waste treatment processes remove contaminants before discharge into environment.",
        "Weather radar networks track precipitation patterns providing early flood warning capabilities.",
        "Healthcare electronic prescribing reduces medication errors through automated verification systems.",
        "Professional credential verification confirms qualifications of licensed practitioners in regulated fields.",
        "Urban historic preservation maintains architectural heritage while accommodating modern development needs.",
        "Scientific grant funding supports investigator-initiated research addressing fundamental questions.",
        "Transportation traffic modeling predicts congestion patterns informing infrastructure investment decisions.",
        "Energy grid reliability ensures consistent electricity supply meeting consumer and business demands.",
        "International trade policy negotiations establish rules governing commercial exchanges between nations.",
        "Academic admissions processes select qualified candidates for limited educational program positions.",
        "Public security measures protect critical infrastructure from physical and cyber threats.",
        "Financial portfolio management optimizes investment returns while managing risk exposure levels.",
        "Environmental species protection laws prevent extinction of threatened plant and animal populations.",
        "Technology software development follows methodologies ensuring code quality and system reliability.",
        "Legal contract enforcement mechanisms provide remedies for breach of agreement obligations.",
        "Manufacturing robotics integration improves precision and consistency in production operations.",
        "Marine fisheries management establishes catch limits promoting sustainable seafood harvests.",
        "Educational distance learning expands access to courses for students in remote locations.",
        "Community housing assistance helps low-income families afford safe and stable living arrangements.",
        "Pharmaceutical drug approval process evaluates safety and efficacy before market authorization.",
        "Broadcasting programming diversity reflects varied audience interests and cultural perspectives.",
        "Financial taxation policy balances revenue generation with economic growth considerations.",
        "Forestry reforestation efforts restore degraded lands improving carbon storage capacity.",
        "Public health vaccination programs prevent communicable disease transmission protecting populations.",
        "Research animal care standards ensure humane treatment of subjects in scientific studies.",
        "Industrial pollution control equipment reduces emissions of harmful substances into atmosphere.",
        "Weather forecasting accuracy improvements result from enhanced modeling and observation systems.",
        "Healthcare telemedicine expansion connects patients with providers overcoming geographic barriers.",
        "Professional continuing education maintains practitioner competence in rapidly evolving fields.",
        "Urban planning zoning regulations guide land use development patterns and building standards.",
        "Scientific open access publishing increases availability of research findings to global community.",
        "Transportation emissions standards limit vehicle pollutants improving air quality in urban areas."
    ]
    
    return additional_real_news


def create_1000_row_dataset():
    """Create dataset with 1000+ rows"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    print("=" * 80)
    print("CREATING 1000+ ROW DATASET")
    print("=" * 80)
    
    # Load existing real news
    print(f"\n[1/6] Loading existing real news...")
    balanced_path = os.path.join(data_dir, 'balanced_training_data.csv')
    balanced_df = pd.read_csv(balanced_path)
    existing_real = balanced_df[balanced_df['label'] == 'real']
    print(f"✓ Existing real news: {len(existing_real)}")
    
    # Generate more real news
    print(f"\n[2/6] Generating additional real news...")
    new_real_news = generate_more_real_news()
    new_real_df = pd.DataFrame({'text': new_real_news, 'label': 'real'})
    print(f"✓ Generated {len(new_real_df)} new real news articles")
    
    # Combine all real news
    all_real = pd.concat([existing_real[['text', 'label']], new_real_df], ignore_index=True)
    print(f"✓ Total real news: {len(all_real)}")
    
    # Load fake news
    print(f"\n[3/6] Loading fake news...")
    kaggle_path = os.path.join(data_dir, 'preprocessed_kaggle_data.csv')
    kaggle_df = pd.read_csv(kaggle_path)
    print(f"✓ Available fake news: {len(kaggle_df)}")
    
    # Sample 500 from each to create balanced 1000+ dataset
    n_samples = 500
    real_sample = all_real.sample(n=min(n_samples, len(all_real)), random_state=42)
    fake_sample = kaggle_df.sample(n=n_samples, random_state=42)
    fake_sample['label'] = 'fake'
    
    # Combine
    df = pd.concat([real_sample, fake_sample[['text', 'label']]], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✓ Combined dataset: {len(df)} articles")
    print(f"  - Real: {(df['label']=='real').sum()}")
    print(f"  - Fake: {(df['label']=='fake').sum()}")
    
    # Super clean
    print(f"\n[4/6] SUPER CLEANING all articles...")
    print("  - This may take a few minutes...")
    
    stop_words = set(stopwords.words('english'))
    extra_stopwords = {
        'of', 'to', 'like', 'we', 'us', 'all', 'also', 'would', 'could', 
        'should', 'may', 'might', 'must', 'will', 'shall', 'can',
        'get', 'got', 'getting', 'go', 'going', 'went', 'gone',
        'say', 'said', 'saying', 'says', 'tell', 'told', 'telling',
        'make', 'made', 'making', 'makes', 'take', 'took', 'taking',
        'know', 'knew', 'known', 'knowing', 'think', 'thought', 'thinking',
        'see', 'saw', 'seen', 'seeing', 'come', 'came', 'coming',
        'want', 'wanted', 'wanting', 'use', 'used', 'using',
        'find', 'found', 'finding', 'give', 'gave', 'given', 'giving',
        'work', 'worked', 'working', 'call', 'called', 'calling',
        'try', 'tried', 'trying', 'ask', 'asked', 'asking',
        'need', 'needed', 'needing', 'feel', 'felt', 'feeling',
        'become', 'became', 'becoming', 'leave', 'left', 'leaving',
        'put', 'putting', 'mean', 'meant', 'meaning', 'keep', 'kept', 'keeping',
        'let', 'letting', 'begin', 'began', 'begun', 'beginning',
        'seem', 'seemed', 'seeming', 'help', 'helped', 'helping',
        'talk', 'talked', 'talking', 'turn', 'turned', 'turning',
        'start', 'started', 'starting', 'show', 'showed', 'shown', 'showing',
        'hear', 'heard', 'hearing', 'play', 'played', 'playing',
        'run', 'ran', 'running', 'move', 'moved', 'moving',
        'live', 'lived', 'living', 'believe', 'believed', 'believing',
        'bring', 'brought', 'bringing', 'happen', 'happened', 'happening',
        'one', 'two', 'three', 'first', 'second', 'third', 'last', 'next',
        'much', 'many', 'more', 'most', 'some', 'any', 'every', 'each',
        'own', 'well', 'even', 'still', 'just', 'back', 'way', 'now'
    }
    stop_words.update(extra_stopwords)
    
    preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)
    
    cleaned_texts = []
    for i, text in enumerate(df['text']):
        if (i + 1) % 100 == 0:
            print(f"  - Progress: {i+1}/{len(df)}")
        
        cleaned = preprocessor.preprocess(text)
        words = cleaned.split()
        words = [w for w in words if w not in stop_words and len(w) > 2]
        cleaned_texts.append(' '.join(words))
    
    df['cleaned_text'] = cleaned_texts
    
    # Remove empty
    print(f"\n[5/6] Removing empty/short texts...")
    df['word_count'] = df['cleaned_text'].str.split().str.len()
    df = df[df['word_count'] >= 5]
    df = df.drop('word_count', axis=1)
    
    print(f"✓ Final: {len(df)} articles")
    
    # Save
    print(f"\n[6/6] Saving...")
    final_df = df[['text', 'cleaned_text', 'label']]
    output_path = os.path.join(data_dir, 'dataset_1000_plus.csv')
    final_df.to_csv(output_path, index=False)
    
    print(f"✓ Saved to: {output_path}")
    
    # Stats
    print(f"\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print(f"  Total: {len(final_df)}")
    print(f"  Real: {(final_df['label']=='real').sum()}")
    print(f"  Fake: {(final_df['label']=='fake').sum()}")
    print(f"  Balance: {(final_df['label']=='real').sum() / len(final_df) * 100:.1f}% real")
    
    print(f"\n" + "=" * 80)
    print("✅ DONE! Train with:")
    print(f"  python src/quick_train_improved.py --data data/dataset_1000_plus.csv")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    create_1000_row_dataset()
