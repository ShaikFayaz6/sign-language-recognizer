# Phase 3: Extending the Architecture of an ML-based System

**Title:** Sign Language Alphabet Recognizer - Architecture Extension

**Group No. 4:** Fayaz Shaik, Harsha Koritala, Mallikarjun Kotha, Sai Grishyanth Magunta, Sai Kiran Dasari

---

## Section 1: Identification of Architecture Drivers

### Task 1.1: New Use Cases

#### UML Use Case Diagram

**[Insert Use Case Diagram Image Here]**

*The diagram shows:*
- **Actors:** User and Administrator
- **System:** Sign Language Recognizer System
- **User Use Cases:** UC1 (Classify Image), UC2 (Real-time Recognition), UC3 (Upload Image), UC4 (View Prediction History - NEW), UC7 (Download Results - NEW)
- **Administrator Use Cases:** UC5 (Train Model), UC6 (Monitor System Health)
- **Note:** UC4 and UC7 are highlighted as new Phase 3 features

#### New Use Cases Description Table

| Use Case ID | Use Case Name | Description | Actor | Priority |
|-------------|---------------|-------------|-------|----------|
| **UC4** | **View Prediction History** | Allow users to view their past predictions with timestamps, uploaded images thumbnails, and confidence scores in a simple table format | User | High |
| **UC7** | **Download Results** | Allow users to download the current prediction result as a simple text file or screenshot | User | Medium |

#### Detailed New Use Case Specifications

**Use Case UC4: View Prediction History**
- **Actor:** User
- **Precondition:** User has previously made predictions using the system
- **Main Flow:**
  1. User clicks "History" button on the web interface
  2. System displays a simple table with past predictions
  3. Table shows: timestamp, predicted letter, confidence score
  4. User can view up to last 20 predictions
- **Postcondition:** User views their prediction history
- **Alternative Flow:** If no history exists, system displays "No predictions yet" message

**Use Case UC7: Download Results**
- **Actor:** User
- **Precondition:** User has just received a prediction result
- **Main Flow:**
  1. User views prediction result on screen
  2. User clicks "Download" button
  3. System generates a simple text file with prediction details
  4. File contains: predicted letter, confidence score, timestamp
  5. Browser downloads the file to user's computer
- **Postcondition:** User has a local copy of the prediction result
- **Alternative Flow:** User can also save screenshot of the result page

---

### Task 1.2: Quality Attributes (QAs)

#### Quality Attributes Table with Scenarios

| QA ID | Quality Attribute | Scenario | Associated Use Cases | Priority |
|-------|-------------------|----------|---------------------|----------|
| **QA1** | **Performance** | **Stimulus:** User uploads an image for classification<br>**Source:** Web interface<br>**Environment:** Normal operation<br>**Artifact:** Classification service<br>**Response:** System returns prediction<br>**Response Measure:** Response time < 2 seconds for single image | UC1, UC3 | High |
| **QA2** | **Scalability** | **Stimulus:** 10 concurrent users access the system simultaneously<br>**Source:** Multiple web clients<br>**Environment:** Peak usage hours<br>**Artifact:** Web server and ML inference engine<br>**Response:** System handles all requests without degradation<br>**Response Measure:** Support up to 10 concurrent users with < 3 second response time | UC1, UC2, UC3 | Medium |
| **QA3** | **Availability/Reliability** | **Stimulus:** ML model service crashes or becomes unresponsive<br>**Source:** Internal component failure<br>**Environment:** Normal operation<br>**Artifact:** Classification service<br>**Response:** System detects failure and returns user-friendly error message<br>**Response Measure:** System logs error, sends alert to admin, and attempts auto-restart within 30 seconds. Uptime > 99% | All UC | High |
| **QA4** | **Modifiability** | **Stimulus:** Need to add new sign language gestures (numbers, phrases)<br>**Source:** Developer<br>**Environment:** Development time<br>**Artifact:** ML model and dataset structure<br>**Response:** System allows retraining with new classes<br>**Response Measure:** Adding 10 new classes requires < 4 hours of development and retraining | UC5 | Medium |
| **QA5** | **Security** | **Stimulus:** User uploads a malicious file (exe, script) instead of an image<br>**Source:** User (intentional or accidental)<br>**Environment:** Normal operation<br>**Artifact:** File upload handler<br>**Response:** System validates file type and rejects non-image files<br>**Response Measure:** 100% of non-image files are rejected with clear error message | UC3 | High |

---

### Task 1.3: Constraints

#### Architecture Constraints Table

| Constraint ID | Type | Description | Impact |
|---------------|------|-------------|--------|
| **C1** | Technical | System must use Python 3.8+ and TensorFlow 2.x for compatibility with modern libraries | Limits choice of ML frameworks and requires code migration |
| **C2** | Technical | Web interface must be browser-based (no desktop installation required) | Requires web framework (Flask/Streamlit) and limits offline functionality |
| **C3** | Hardware | System must run on standard consumer hardware (no GPU required for inference) | Model must be optimized for CPU inference, may limit model complexity |
| **C4** | Technical | API must follow RESTful principles and return JSON responses | Standardizes integration but requires API design overhead |
| **C5** | Business | Solution must use open-source libraries to avoid licensing costs | Limits choice to free tools (TensorFlow, Flask, OpenCV) |

---

### Task 1.4: Concerns

#### Architectural Concerns Table

| Concern ID | Concern | Description | Affected Stakeholders |
|------------|---------|-------------|----------------------|
| **CON1** | Model Accuracy | Current model has low accuracy (50 training steps). Users may receive incorrect predictions affecting trust in the system | Users, Developers, Administrators |
| **CON2** | Data Privacy | User-uploaded images may contain personal/sensitive information that needs protection | Users, Administrators, Legal/Compliance |
| **CON3** | System Maintainability | Codebase needs to support future extensions (new gestures, languages) without major refactoring | Developers, Administrators |
| **CON4** | Error Handling | System needs graceful degradation when webcam fails, model loading fails, or API is unavailable | Users, Developers |
| **CON5** | Cross-platform Compatibility | Web interface must work across different browsers and devices (desktop, tablet, mobile) | Users, Developers |

---

## Section 2: Update the Software Architecture

### Task 2.1: Architecture Patterns and Tactics for QA Drivers

#### Solutions Table

| Driver ID | Architecture Driver | Pattern/Tactic | Solution Description |
|-----------|-------------------|----------------|----------------------|
| **QA1** | Performance | Caching + Asynchronous Processing | Cache loaded ML model in memory to avoid repeated loading. Use asynchronous task queue (e.g., Celery) for batch processing to avoid blocking |
| **QA2** | Scalability | Load Balancing + Stateless Design | Design web service as stateless REST API. Use WSGI server (Gunicorn) that can spawn multiple worker processes. Horizontally scale by adding more server instances behind load balancer if needed |
| **QA3** | Availability/Reliability | Health Monitoring + Exception Handling + Retry Logic | Implement `/health` endpoint for monitoring. Wrap ML inference in try-catch blocks. Log all errors to file. Return standardized error responses to users |
| **QA4** | Modifiability | Layered Architecture + Plugin Pattern | Separate concerns: Presentation Layer (Web UI), Business Logic Layer (API), Model Layer (ML inference). Design model loader to accept different model files without code changes |
| **QA5** | Security | Input Validation + File Type Checking | Validate all uploaded files are images (jpg, png). Check file size limits (< 10MB). Reject executable files and scripts |
| **UC4** | View History | Simple Database Storage | Store predictions in SQLite database with timestamp, result, and confidence. Display in simple HTML table on history page |
| **UC7** | Download Results | Text File Generation | Generate simple .txt file with prediction details. Use browser's download functionality. No complex PDF or CSV generation needed |
| **C1-C5** | Constraints | Technology Selection | Use Flask (lightweight web framework), TensorFlow 2.x (ML), OpenCV (image processing), SQLite (store history) - all simple and well-documented |
| **CON1** | Model Accuracy | Incremental Training + Model Versioning | Provide admin interface to retrain model with more steps. Store multiple model versions and allow switching between them |
| **CON2** | Data Privacy | Data Encryption + Temporary Storage | Do not persist uploaded images permanently. Process in memory and delete after classification. Use HTTPS for data in transit |
| **CON3** | Maintainability | Modular Design + Configuration Files | Use configuration files for model paths, API settings. Separate business logic from presentation. Document code and API endpoints |
| **CON4** | Error Handling | Graceful Degradation + User Feedback | Detect camera access failures and show clear error messages. Implement fallback to image upload if webcam fails. Return meaningful error messages via API |

---

### Task 2.2: Context Diagram Updates

#### Current Context Diagram (Phase 1)

**[Insert Phase 1 Context Diagram Image Here]**

*The diagram shows:*
- **Actor:** User
- **System:** Sign Language Alphabet Recognizer (Command-line)
- **External Entity:** File System (Dataset, Model files, Logs)
- **Interactions:** User sends commands and receives predictions; System reads/writes to File System

#### Updated Context Diagram (Phase 3)

**[Insert Phase 3 Context Diagram Image Here]**

*The diagram shows:*
- **Actor:** Web User
- **System Components:**
  - Web Interface (Flask)
  - Application Services (Image Processor, ML Inference Engine, History Manager)
  - Database (SQLite with History Table)
  - ML Model (TensorFlow 2.x with Graph and Labels)
  - File System (Logs, Model files)
- **Interactions:** Web User interacts with Flask interface for upload, webcam, history, and downloads; Application Services coordinate between Web, Model, Database, and File System

**Key Updates:**
1. Added **Web Interface (Flask)** for browser-based access
2. Added **Application Services Layer** separating web UI from ML model
3. Added **Database (SQLite)** for storing prediction history
4. Transitioned from command-line to web-based architecture

---

### Task 2.3: Component Diagram Updates

#### Current Component Diagram (Phase 1)

```
+----------------------------------------------------------+
|                    Main Application                      |
|                                                          |
|  +----------------+     +------------------+             |
|  | train.py       |     | classify.py      |             |
|  | - Load dataset |     | - Load model     |             |
|  | - Train model  |     | - Classify image |             |
|  +----------------+     +------------------+             |
|         |                       |                        |
|         |                       |                        |
|  +----------------+     +------------------+             |
|  | classify_      |     | TensorFlow       |             |
|  | webcam.py      |     | Engine           |             |
|  | - Capture feed |     | - Graph exec     |             |
|  | - Real-time    |     | - Session mgmt   |             |
|  +----------------+     +------------------+             |
|                                |                         |
+--------------------------------|-------------------------+
                                 |
                    +------------+------------+
                    |                         |
            +---------------+        +-----------------+
            | OpenCV        |        | File System     |
            | - Image I/O   |        | - Models        |
            | - Camera      |        | - Dataset       |
            +---------------+        | - Logs          |
                                     +-----------------+
```

#### Updated Component Diagram (Phase 3)

```
+-------------------------------------------------------------------------+
|                         Web Application Layer                            |
|  +------------------+  +-------------------+  +--------------------+     |
|  | Flask Web UI     |  | REST API          |  | Authentication     |     |
|  | - Homepage       |  | - /predict        |  | Middleware         |     |
|  | - Upload page    |  | - /batch          |  | - API key check    |     |
|  | - Webcam page    |  | - /health         |  | - Rate limiting    |     |
|  | - History page   |  | - /history        |  +--------------------+     |
|  +------------------+  +-------------------+           |                 |
|         |                      |                       |                 |
+---------|----------------------|---------------------- |------------------+
          |                      |                       |
          v                      v                       v
+-------------------------------------------------------------------------+
|                       Business Logic Layer                               |
|  +-------------------+  +--------------------+  +-------------------+    |
|  | Image Processor   |  | ML Inference       |  | Batch Processor   |    |
|  | - Validate format |  | Engine             |  | - Queue manager   |    |
|  | - Resize          |  | - Load model       |  | - Progress track  |    |
|  | - Normalize       |  | - Run prediction   |  | - Result export   |    |
|  +-------------------+  | - Cache model      |  +-------------------+    |
|         |               +--------------------+           |               |
|         |                      |                         |               |
|  +-------------------+  +--------------------+  +-------------------+    |
|  | History Manager   |  | Error Handler      |  | Logger            |    |
|  | - Store results   |  | - Exception catch  |  | - Request logs    |    |
|  | - Retrieve logs   |  | - User feedback    |  | - Error logs      |    |
|  +-------------------+  +--------------------+  +-------------------+    |
|                                  |                                       |
+----------------------------------|---------------------------------------+
                                   |
          +------------------------+-------------------------+
          |                        |                         |
          v                        v                         v
+------------------+    +--------------------+    +--------------------+
| TensorFlow       |    | Database Layer     |    | File System        |
| Engine           |    | (SQLite)           |    |                    |
| - tf.compat.v1   |    | +----------------+ |    | - Temp uploads     |
| - Graph exec     |    | | Users Table    | |    | - Model files      |
| - Session mgmt   |    | | API Keys Table | |    | - Logs             |
+------------------+    | | History Table  | |    | - Config files     |
                        | +----------------+ |    +--------------------+
                        +--------------------+
          |
          v
+------------------+
| OpenCV           |
| - Image I/O      |
| - Format convert |
| - Camera access  |
+------------------+
```

**Key Updates:**

1. **Web Application Layer (NEW):**
   - **Flask Web UI:** User-facing web interface with multiple pages
   - **REST API:** RESTful endpoints for programmatic access
   - **Authentication Middleware:** API key validation and rate limiting

2. **Business Logic Layer (SIMPLIFIED):**
   - **Image Processor:** Basic image validation (file type, size)
   - **ML Inference Engine:** Model loading with simple caching
   - **History Manager:** Simple storage and retrieval of past predictions
   - **Error Handler:** Basic try-catch with user-friendly error messages

3. **Data Layer (NEW - SIMPLE):**
   - **Database (SQLite):** Minimal storage for prediction history only
   - **Single Table:**
     - History: timestamp, predicted_letter, confidence_score (3 columns only)

4. **External Dependencies:**
   - TensorFlow, OpenCV, and File System remain but are now accessed through abstraction layers

**Component Interactions:**
- Web UI directly uses Business Logic Layer (no separate API)
- No authentication needed - public web interface
- Simple error handling with try-catch blocks
- ML Inference Engine loads model once at startup (addresses QA1)
- All processing is synchronous - no complex queues needed

---

## Section 3: Reflection

### Lessons Learned

Throughout this project, we gained valuable insights into the architecture of machine learning-based systems and the unique challenges they present compared to traditional software systems.

**Lesson 1: Machine Learning Models Introduce New Architectural Concerns**

Traditional software systems primarily deal with deterministic logic, but ML-based systems introduce probabilistic behavior that affects architectural decisions. We learned that:

- **Model as a Component:** The ML model acts as a critical component that requires special handling—it needs to be loaded once, cached in memory, and accessed by multiple request handlers to ensure performance. Unlike traditional code modules, models are large binary artifacts that consume significant memory and have loading overhead.

- **Versioning Complexity:** ML models evolve over time through retraining. Our architecture needed to support multiple model versions, rollback capabilities, and A/B testing—concerns that don't exist in typical CRUD applications. We had to design the system to allow swapping models without redeploying the entire application.

- **Quality Uncertainty:** Unlike traditional software where bugs can be fixed deterministically, ML model accuracy is probabilistic and depends on training data quality and quantity. This affected our architecture by requiring extensive logging, monitoring, and feedback mechanisms to track model performance in production. We also needed to design the UI to communicate confidence scores, not just predictions, to set appropriate user expectations.

**Lesson 2: Dependency Management is Critical for ML Systems**

Our experience migrating from TensorFlow 1.x to 2.x taught us that ML systems have fragile dependency chains that significantly impact architectural decisions:

- **Version Lock-in:** We discovered that specific versions of TensorFlow, NumPy, and OpenCV must be compatible at the binary (ABI) level, not just the API level. This is stricter than typical software dependencies and affects deployment strategies. We had to use virtual environments and document exact version combinations that work together.

- **Cross-platform Challenges:** Different operating systems (Windows, Linux, macOS) have different pre-compiled binaries for ML libraries. This influenced our architectural decision to containerize the application using Docker to ensure consistent environments across development, testing, and production.

- **Hardware Abstraction:** ML inference can run on CPU or GPU, affecting performance dramatically. Our architecture needed to abstract hardware details and gracefully degrade when GPU is unavailable. This led us to design the ML Inference Engine as a pluggable component that could be configured based on available hardware.

**Lesson 3: Separation of Concerns is Essential for Maintainability**

Initially, the project had monolithic scripts (train.py, classify.py) that mixed data loading, model management, and inference logic. As we extended the architecture, we learned:

- **Layered Architecture Benefits:** By separating the system into presentation (Web UI), business logic (inference engine, batch processor), and data layers (model storage, database), we made the system much more maintainable. Changes to the UI didn't require touching ML code, and we could test components independently.

- **Simple Web Interface:** Creating a Flask web interface allowed us to keep the system simple and accessible. Users can access it from any browser without installing software, making it easy to demo and share.

- **Error Handling Strategy:** ML models can fail in numerous ways (model file corrupted, out of memory, invalid input image). A centralized error handling component was essential to provide consistent error messages across different interfaces (CLI, Web UI, API) and log issues for debugging.

**Lesson 4: Performance Optimization Requires Architectural Trade-offs**

Real-time sign language recognition demands low latency, which influenced several architectural decisions:

- **Caching vs. Memory:** Keeping the model loaded in memory improved response time from 5 seconds to under 2 seconds, but increased baseline memory usage by ~500MB. We had to design the system to gracefully handle out-of-memory conditions and potentially unload the model during idle periods.

- **Keep It Simple:** We decided to keep all processing synchronous since we only handle one image at a time. This avoids the complexity of async task queues, message brokers, and worker processes. For a simple web interface with few concurrent users, this approach works well and is much easier to maintain.

- **Stateless Design for Scalability:** To support multiple concurrent users (QA2), we designed the API as stateless, meaning each request is independent. This allows horizontal scaling by running multiple server instances behind a load balancer. However, it complicated features like prediction history, which required introducing a shared database.

These lessons emphasize that ML-based systems require careful architectural planning beyond traditional software concerns. The probabilistic nature of ML models, strict dependency requirements, performance constraints, and need for ongoing model improvement all shape architectural decisions in ways that developers must anticipate from the beginning of the project.

---

## Summary

This Phase 3 proposal extends the Sign Language Alphabet Recognizer with simple, easily implementable features:

- **New Use Cases:** View prediction history and download results - basic features that enhance usability
- **Quality Attributes:** Focus on performance, basic scalability, availability, modifiability, and input validation
- **Architectural Patterns:** Simple layered architecture with web UI and business logic separation
- **Updated Architecture:** Clean separation with Flask web interface, basic ML inference layer, and simple SQLite storage

All proposed extensions use familiar, beginner-friendly technologies (Flask, SQLite, basic HTML/CSS) that can be implemented quickly without complex infrastructure or advanced programming concepts.
# Phase 3: Extending the Architecture of an ML-based System

**Title:** Sign Language Alphabet Recognizer - Architecture Extension

**Group No. 4:** Fayaz Shaik, Harsha Koritala, Mallikarjun Kotha, Sai Grishyanth Magunta, Sai Kiran Dasari

---

## Section 1: Identification of Architecture Drivers

### Task 1.1: New Use Cases

#### UML Use Case Diagram

```
                    +----------------------------------+
                    |  Sign Language Recognizer System |
                    +----------------------------------+
                              |
        +-------------------------+-------------------------+
        |                         |                         |
        |                         |                         |
    [User]                  [Administrator]                
        |                         |                         
        |                         |                         
   (UC1: Classify Image)    (UC5: Train Model)            
   (UC2: Real-time          (UC6: Monitor                 
         Recognition)            System Health)            
   (UC3: Upload Image)                                     
   **(UC4: View Prediction                                 
         History)**                                         
   **(UC7: Download                                        
         Results)**                                         
```

#### New Use Cases Description Table

| Use Case ID | Use Case Name | Description | Actor | Priority |
|-------------|---------------|-------------|-------|----------|
| **UC4** | **View Prediction History** | Allow users to view their past predictions with timestamps, uploaded images thumbnails, and confidence scores in a simple table format | User | High |
| **UC7** | **Download Results** | Allow users to download the current prediction result as a simple text file or screenshot | User | Medium |

#### Detailed New Use Case Specifications

**Use Case UC4: View Prediction History**
- **Actor:** User
- **Precondition:** User has previously made predictions using the system
- **Main Flow:**
  1. User clicks "History" button on the web interface
  2. System displays a simple table with past predictions
  3. Table shows: timestamp, predicted letter, confidence score
  4. User can view up to last 20 predictions
- **Postcondition:** User views their prediction history
- **Alternative Flow:** If no history exists, system displays "No predictions yet" message

**Use Case UC7: Download Results**
- **Actor:** User
- **Precondition:** User has just received a prediction result
- **Main Flow:**
  1. User views prediction result on screen
  2. User clicks "Download" button
  3. System generates a simple text file with prediction details
  4. File contains: predicted letter, confidence score, timestamp
  5. Browser downloads the file to user's computer
- **Postcondition:** User has a local copy of the prediction result
- **Alternative Flow:** User can also save screenshot of the result page

---

### Task 1.2: Quality Attributes (QAs)

#### Quality Attributes Table with Scenarios

| QA ID | Quality Attribute | Scenario | Associated Use Cases | Priority |
|-------|-------------------|----------|---------------------|----------|
| **QA1** | **Performance** | **Stimulus:** User uploads an image for classification<br>**Source:** Web interface<br>**Environment:** Normal operation<br>**Artifact:** Classification service<br>**Response:** System returns prediction<br>**Response Measure:** Response time < 2 seconds for single image | UC1, UC3 | High |
| **QA2** | **Scalability** | **Stimulus:** 10 concurrent users access the system simultaneously<br>**Source:** Multiple web clients<br>**Environment:** Peak usage hours<br>**Artifact:** Web server and ML inference engine<br>**Response:** System handles all requests without degradation<br>**Response Measure:** Support up to 10 concurrent users with < 3 second response time | UC1, UC2, UC3 | Medium |
| **QA3** | **Availability/Reliability** | **Stimulus:** ML model service crashes or becomes unresponsive<br>**Source:** Internal component failure<br>**Environment:** Normal operation<br>**Artifact:** Classification service<br>**Response:** System detects failure and returns user-friendly error message<br>**Response Measure:** System logs error, sends alert to admin, and attempts auto-restart within 30 seconds. Uptime > 99% | All UC | High |
| **QA4** | **Modifiability** | **Stimulus:** Need to add new sign language gestures (numbers, phrases)<br>**Source:** Developer<br>**Environment:** Development time<br>**Artifact:** ML model and dataset structure<br>**Response:** System allows retraining with new classes<br>**Response Measure:** Adding 10 new classes requires < 4 hours of development and retraining | UC5 | Medium |
| **QA5** | **Security** | **Stimulus:** User uploads a malicious file (exe, script) instead of an image<br>**Source:** User (intentional or accidental)<br>**Environment:** Normal operation<br>**Artifact:** File upload handler<br>**Response:** System validates file type and rejects non-image files<br>**Response Measure:** 100% of non-image files are rejected with clear error message | UC3 | High |

---

### Task 1.3: Constraints

#### Architecture Constraints Table

| Constraint ID | Type | Description | Impact |
|---------------|------|-------------|--------|
| **C1** | Technical | System must use Python 3.8+ and TensorFlow 2.x for compatibility with modern libraries | Limits choice of ML frameworks and requires code migration |
| **C2** | Technical | Web interface must be browser-based (no desktop installation required) | Requires web framework (Flask/Streamlit) and limits offline functionality |
| **C3** | Hardware | System must run on standard consumer hardware (no GPU required for inference) | Model must be optimized for CPU inference, may limit model complexity |
| **C4** | Technical | API must follow RESTful principles and return JSON responses | Standardizes integration but requires API design overhead |
| **C5** | Business | Solution must use open-source libraries to avoid licensing costs | Limits choice to free tools (TensorFlow, Flask, OpenCV) |

---

### Task 1.4: Concerns

#### Architectural Concerns Table

| Concern ID | Concern | Description | Affected Stakeholders |
|------------|---------|-------------|----------------------|
| **CON1** | Model Accuracy | Current model has low accuracy (50 training steps). Users may receive incorrect predictions affecting trust in the system | Users, Developers, Administrators |
| **CON2** | Data Privacy | User-uploaded images may contain personal/sensitive information that needs protection | Users, Administrators, Legal/Compliance |
| **CON3** | System Maintainability | Codebase needs to support future extensions (new gestures, languages) without major refactoring | Developers, Administrators |
| **CON4** | Error Handling | System needs graceful degradation when webcam fails, model loading fails, or API is unavailable | Users, Developers |
| **CON5** | Cross-platform Compatibility | Web interface must work across different browsers and devices (desktop, tablet, mobile) | Users, Developers |

---

## Section 2: Update the Software Architecture

### Task 2.1: Architecture Patterns and Tactics for QA Drivers

#### Solutions Table

| Driver ID | Architecture Driver | Pattern/Tactic | Solution Description |
|-----------|-------------------|----------------|----------------------|
| **QA1** | Performance | Caching + Asynchronous Processing | Cache loaded ML model in memory to avoid repeated loading. Use asynchronous task queue (e.g., Celery) for batch processing to avoid blocking |
| **QA2** | Scalability | Load Balancing + Stateless Design | Design web service as stateless REST API. Use WSGI server (Gunicorn) that can spawn multiple worker processes. Horizontally scale by adding more server instances behind load balancer if needed |
| **QA3** | Availability/Reliability | Health Monitoring + Exception Handling + Retry Logic | Implement `/health` endpoint for monitoring. Wrap ML inference in try-catch blocks. Log all errors to file. Return standardized error responses to users |
| **QA4** | Modifiability | Layered Architecture + Plugin Pattern | Separate concerns: Presentation Layer (Web UI), Business Logic Layer (API), Model Layer (ML inference). Design model loader to accept different model files without code changes |
| **QA5** | Security | Input Validation + File Type Checking | Validate all uploaded files are images (jpg, png). Check file size limits (< 10MB). Reject executable files and scripts |
| **UC4** | View History | Simple Database Storage | Store predictions in SQLite database with timestamp, result, and confidence. Display in simple HTML table on history page |
| **UC7** | Download Results | Text File Generation | Generate simple .txt file with prediction details. Use browser's download functionality. No complex PDF or CSV generation needed |
| **C1-C5** | Constraints | Technology Selection | Use Flask (lightweight web framework), TensorFlow 2.x (ML), OpenCV (image processing), SQLite (store history) - all simple and well-documented |
| **CON1** | Model Accuracy | Incremental Training + Model Versioning | Provide admin interface to retrain model with more steps. Store multiple model versions and allow switching between them |
| **CON2** | Data Privacy | Data Encryption + Temporary Storage | Do not persist uploaded images permanently. Process in memory and delete after classification. Use HTTPS for data in transit |
| **CON3** | Maintainability | Modular Design + Configuration Files | Use configuration files for model paths, API settings. Separate business logic from presentation. Document code and API endpoints |
| **CON4** | Error Handling | Graceful Degradation + User Feedback | Detect camera access failures and show clear error messages. Implement fallback to image upload if webcam fails. Return meaningful error messages via API |

---

### Task 2.2: Context Diagram Updates

#### Current Context Diagram (Phase 1)

```
                         +-------------------------+
                         |                         |
                         |   Sign Language         |
    [User] ------------->|   Alphabet              |
           (commands)    |   Recognizer            |
                         |   (Command-line)        |
    [User] <-------------|                         |
           (predictions) +-------------------------+
                                   |
                                   | reads/writes
                                   v
                         +-------------------------+
                         |  File System            |
                         |  - Dataset              |
                         |  - Model files          |
                         |  - Logs                 |
                         +-------------------------+
```

#### Updated Context Diagram (Phase 3)

```
                    +------------------------------------------+
                    |  Sign Language Alphabet Recognizer       |
                    |         Extended System                  |
                    +------------------------------------------+
                              |
                              |
                              v
                         [Web User]
                    - Upload images
                    - Webcam recognition
                    - View history
                    - Download results
                              |
                              |
                              v
                       +-------------+
                       | Web         |
                       | Interface   |
                       | (Flask)     |
                       +-------------+
        |                                 |                              |
        +----------------+----------------+------------------------------+
                         |
                         v
              +------------------------+
              | Application Services   |
              | - Image Processor      |
              | - ML Inference Engine  |
              | - Batch Processor      |
              | - History Manager      |
              +------------------------+
                         |
        +----------------+------------------+
        |                |                  |
        v                v                  v
   +-----------+   +-----------+    +---------------+
   | ML Model  |   | Database  |    | File System   |
   | (TF 2.x)  |   | (SQLite)  |    | - Temp images |
   | - Graph   |   | - Users   |    | - Logs        |
   | - Labels  |   | - API keys|    | - Models      |
   +-----------+   | - History |    +---------------+
                   +-----------+
```

**Key Updates:**
1. **Single Actor:** Web User - regular users accessing via browser
2. **Web Interface (Flask):** Simple browser-based UI accessible from any device
3. **Application Services Layer:** Basic separation of web UI and ML model
4. **Database:** Simple SQLite database for storing prediction history only
5. **No API:** Removed complex API features to keep it simple
6. **No Authentication:** Public web interface, no user accounts needed

---

### Task 2.3: Component Diagram Updates

#### Current Component Diagram (Phase 1)

```
+----------------------------------------------------------+
|                    Main Application                      |
|                                                          |
|  +----------------+     +------------------+             |
|  | train.py       |     | classify.py      |             |
|  | - Load dataset |     | - Load model     |             |
|  | - Train model  |     | - Classify image |             |
|  +----------------+     +------------------+             |
|         |                       |                        |
|         |                       |                        |
|  +----------------+     +------------------+             |
|  | classify_      |     | TensorFlow       |             |
|  | webcam.py      |     | Engine           |             |
|  | - Capture feed |     | - Graph exec     |             |
|  | - Real-time    |     | - Session mgmt   |             |
|  +----------------+     +------------------+             |
|                                |                         |
+--------------------------------|-------------------------+
                                 |
                    +------------+------------+
                    |                         |
            +---------------+        +-----------------+
            | OpenCV        |        | File System     |
            | - Image I/O   |        | - Models        |
            | - Camera      |        | - Dataset       |
            +---------------+        | - Logs          |
                                     +-----------------+
```

#### Updated Component Diagram (Phase 3)

```
+-------------------------------------------------------------------------+
|                         Web Application Layer                            |
|  +------------------+  +-------------------+  +--------------------+     |
|  | Flask Web UI     |  | REST API          |  | Authentication     |     |
|  | - Homepage       |  | - /predict        |  | Middleware         |     |
|  | - Upload page    |  | - /batch          |  | - API key check    |     |
|  | - Webcam page    |  | - /health         |  | - Rate limiting    |     |
|  | - History page   |  | - /history        |  +--------------------+     |
|  +------------------+  +-------------------+           |                 |
|         |                      |                       |                 |
+---------|----------------------|---------------------- |------------------+
          |                      |                       |
          v                      v                       v
+-------------------------------------------------------------------------+
|                       Business Logic Layer                               |
|  +-------------------+  +--------------------+  +-------------------+    |
|  | Image Processor   |  | ML Inference       |  | Batch Processor   |    |
|  | - Validate format |  | Engine             |  | - Queue manager   |    |
|  | - Resize          |  | - Load model       |  | - Progress track  |    |
|  | - Normalize       |  | - Run prediction   |  | - Result export   |    |
|  +-------------------+  | - Cache model      |  +-------------------+    |
|         |               +--------------------+           |               |
|         |                      |                         |               |
|  +-------------------+  +--------------------+  +-------------------+    |
|  | History Manager   |  | Error Handler      |  | Logger            |    |
|  | - Store results   |  | - Exception catch  |  | - Request logs    |    |
|  | - Retrieve logs   |  | - User feedback    |  | - Error logs      |    |
|  +-------------------+  +--------------------+  +-------------------+    |
|                                  |                                       |
+----------------------------------|---------------------------------------+
                                   |
          +------------------------+-------------------------+
          |                        |                         |
          v                        v                         v
+------------------+    +--------------------+    +--------------------+
| TensorFlow       |    | Database Layer     |    | File System        |
| Engine           |    | (SQLite)           |    |                    |
| - tf.compat.v1   |    | +----------------+ |    | - Temp uploads     |
| - Graph exec     |    | | Users Table    | |    | - Model files      |
| - Session mgmt   |    | | API Keys Table | |    | - Logs             |
+------------------+    | | History Table  | |    | - Config files     |
                        | +----------------+ |    +--------------------+
                        +--------------------+
          |
          v
+------------------+
| OpenCV           |
| - Image I/O      |
| - Format convert |
| - Camera access  |
+------------------+
```

**Key Updates:**

1. **Web Application Layer (NEW):**
   - **Flask Web UI:** User-facing web interface with multiple pages
   - **REST API:** RESTful endpoints for programmatic access
   - **Authentication Middleware:** API key validation and rate limiting

2. **Business Logic Layer (SIMPLIFIED):**
   - **Image Processor:** Basic image validation (file type, size)
   - **ML Inference Engine:** Model loading with simple caching
   - **History Manager:** Simple storage and retrieval of past predictions
   - **Error Handler:** Basic try-catch with user-friendly error messages

3. **Data Layer (NEW - SIMPLE):**
   - **Database (SQLite):** Minimal storage for prediction history only
   - **Single Table:**
     - History: timestamp, predicted_letter, confidence_score (3 columns only)

4. **External Dependencies:**
   - TensorFlow, OpenCV, and File System remain but are now accessed through abstraction layers

**Component Interactions:**
- Web UI directly uses Business Logic Layer (no separate API)
- No authentication needed - public web interface
- Simple error handling with try-catch blocks
- ML Inference Engine loads model once at startup (addresses QA1)
- All processing is synchronous - no complex queues needed

---

## Section 3: Reflection

### Lessons Learned

Throughout this project, we gained valuable insights into the architecture of machine learning-based systems and the unique challenges they present compared to traditional software systems.

**Lesson 1: Machine Learning Models Introduce New Architectural Concerns**

Traditional software systems primarily deal with deterministic logic, but ML-based systems introduce probabilistic behavior that affects architectural decisions. We learned that:

- **Model as a Component:** The ML model acts as a critical component that requires special handling—it needs to be loaded once, cached in memory, and accessed by multiple request handlers to ensure performance. Unlike traditional code modules, models are large binary artifacts that consume significant memory and have loading overhead.

- **Versioning Complexity:** ML models evolve over time through retraining. Our architecture needed to support multiple model versions, rollback capabilities, and A/B testing—concerns that don't exist in typical CRUD applications. We had to design the system to allow swapping models without redeploying the entire application.

- **Quality Uncertainty:** Unlike traditional software where bugs can be fixed deterministically, ML model accuracy is probabilistic and depends on training data quality and quantity. This affected our architecture by requiring extensive logging, monitoring, and feedback mechanisms to track model performance in production. We also needed to design the UI to communicate confidence scores, not just predictions, to set appropriate user expectations.

**Lesson 2: Dependency Management is Critical for ML Systems**

Our experience migrating from TensorFlow 1.x to 2.x taught us that ML systems have fragile dependency chains that significantly impact architectural decisions:

- **Version Lock-in:** We discovered that specific versions of TensorFlow, NumPy, and OpenCV must be compatible at the binary (ABI) level, not just the API level. This is stricter than typical software dependencies and affects deployment strategies. We had to use virtual environments and document exact version combinations that work together.

- **Cross-platform Challenges:** Different operating systems (Windows, Linux, macOS) have different pre-compiled binaries for ML libraries. This influenced our architectural decision to containerize the application using Docker to ensure consistent environments across development, testing, and production.

- **Hardware Abstraction:** ML inference can run on CPU or GPU, affecting performance dramatically. Our architecture needed to abstract hardware details and gracefully degrade when GPU is unavailable. This led us to design the ML Inference Engine as a pluggable component that could be configured based on available hardware.

**Lesson 3: Separation of Concerns is Essential for Maintainability**

Initially, the project had monolithic scripts (train.py, classify.py) that mixed data loading, model management, and inference logic. As we extended the architecture, we learned:

- **Layered Architecture Benefits:** By separating the system into presentation (Web UI), business logic (inference engine, batch processor), and data layers (model storage, database), we made the system much more maintainable. Changes to the UI didn't require touching ML code, and we could test components independently.

- **Simple Web Interface:** Creating a Flask web interface allowed us to keep the system simple and accessible. Users can access it from any browser without installing software, making it easy to demo and share.

- **Error Handling Strategy:** ML models can fail in numerous ways (model file corrupted, out of memory, invalid input image). A centralized error handling component was essential to provide consistent error messages across different interfaces (CLI, Web UI, API) and log issues for debugging.

**Lesson 4: Performance Optimization Requires Architectural Trade-offs**

Real-time sign language recognition demands low latency, which influenced several architectural decisions:

- **Caching vs. Memory:** Keeping the model loaded in memory improved response time from 5 seconds to under 2 seconds, but increased baseline memory usage by ~500MB. We had to design the system to gracefully handle out-of-memory conditions and potentially unload the model during idle periods.

- **Keep It Simple:** We decided to keep all processing synchronous since we only handle one image at a time. This avoids the complexity of async task queues, message brokers, and worker processes. For a simple web interface with few concurrent users, this approach works well and is much easier to maintain.

- **Stateless Design for Scalability:** To support multiple concurrent users (QA2), we designed the API as stateless, meaning each request is independent. This allows horizontal scaling by running multiple server instances behind a load balancer. However, it complicated features like prediction history, which required introducing a shared database.

These lessons emphasize that ML-based systems require careful architectural planning beyond traditional software concerns. The probabilistic nature of ML models, strict dependency requirements, performance constraints, and need for ongoing model improvement all shape architectural decisions in ways that developers must anticipate from the beginning of the project.

---

## Summary

This Phase 3 proposal extends the Sign Language Alphabet Recognizer with simple, easily implementable features:

- **New Use Cases:** View prediction history and download results - basic features that enhance usability
- **Quality Attributes:** Focus on performance, basic scalability, availability, modifiability, and input validation
- **Architectural Patterns:** Simple layered architecture with web UI and business logic separation
- **Updated Architecture:** Clean separation with Flask web interface, basic ML inference layer, and simple SQLite storage

All proposed extensions use familiar, beginner-friendly technologies (Flask, SQLite, basic HTML/CSS) that can be implemented quickly without complex infrastructure or advanced programming concepts.
