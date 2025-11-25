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
        +---------------------+----------------------+
        |                     |                      |
        |                     |                      |
    [User]              [Administrator]        [Mobile User]
        |                     |                      |
        |                     |                      |
   (UC1: Train Model)    (UC5: Monitor          (UC7: Translate
   (UC2: Classify Image)      System Health)         Sign via
   (UC3: Real-time           (UC6: View Training     Mobile App)
         Recognition)              Statistics)     (UC8: Save
   (UC4: Upload Image)                                Translation
   **(UC9: Batch Process                              History)*
         Multiple Images)**                         
   **(UC10: API Access
         for Third-party
         Integration)**
```

#### New Use Cases Description Table

| Use Case ID | Use Case Name | Description | Actor | Priority |
|-------------|---------------|-------------|-------|----------|
| **UC9** | **Batch Process Multiple Images** | Allow users to upload multiple images at once and receive predictions for all images in a downloadable report (CSV/JSON format) | User | High |
| **UC10** | **API Access for Third-party Integration** | Provide RESTful API endpoints that allow external applications to send images and receive predictions programmatically | Developer/Third-party Application | High |

#### Detailed New Use Case Specifications

**Use Case UC9: Batch Process Multiple Images**
- **Actor:** User
- **Precondition:** User has multiple sign language images to classify
- **Main Flow:**
  1. User selects "Batch Upload" option
  2. System prompts user to select multiple images
  3. User selects images (2-100 images)
  4. System processes each image sequentially
  5. System displays progress bar
  6. System generates results with predictions and confidence scores
  7. User downloads results as CSV or JSON
- **Postcondition:** All images are classified and results are saved
- **Alternative Flow:** If processing fails for any image, system logs error and continues with remaining images

**Use Case UC10: API Access for Third-party Integration**
- **Actor:** Developer/Third-party Application
- **Precondition:** Developer has API key and endpoint documentation
- **Main Flow:**
  1. External application sends POST request with image data
  2. System validates API key
  3. System processes image
  4. System returns JSON response with prediction and confidence
  5. External application receives and uses the prediction
- **Postcondition:** Prediction is returned to calling application
- **Alternative Flow:** If API key is invalid, return 401 Unauthorized error

---

### Task 1.2: Quality Attributes (QAs)

#### Quality Attributes Table with Scenarios

| QA ID | Quality Attribute | Scenario | Associated Use Cases | Priority |
|-------|-------------------|----------|---------------------|----------|
| **QA1** | **Performance** | **Stimulus:** User uploads an image for classification<br>**Source:** Web interface or API<br>**Environment:** Normal operation<br>**Artifact:** Classification service<br>**Response:** System returns prediction<br>**Response Measure:** Response time < 2 seconds for single image, < 30 seconds for batch of 50 images | UC2, UC4, UC9, UC10 | High |
| **QA2** | **Scalability** | **Stimulus:** 50 concurrent users access the system simultaneously<br>**Source:** Multiple web clients and API calls<br>**Environment:** Peak usage hours<br>**Artifact:** Web server and ML inference engine<br>**Response:** System handles all requests without degradation<br>**Response Measure:** Support up to 100 concurrent users with < 5 second response time | UC3, UC7, UC10 | Medium |
| **QA3** | **Availability/Reliability** | **Stimulus:** ML model service crashes or becomes unresponsive<br>**Source:** Internal component failure<br>**Environment:** Normal operation<br>**Artifact:** Classification service<br>**Response:** System detects failure and returns user-friendly error message<br>**Response Measure:** System logs error, sends alert to admin, and attempts auto-restart within 30 seconds. Uptime > 99% | All UC | High |
| **QA4** | **Modifiability** | **Stimulus:** Need to add new sign language gestures (numbers, phrases)<br>**Source:** Developer<br>**Environment:** Development time<br>**Artifact:** ML model and dataset structure<br>**Response:** System allows retraining with new classes<br>**Response Measure:** Adding 10 new classes requires < 4 hours of development and retraining | UC1 | Medium |
| **QA5** | **Security** | **Stimulus:** Unauthorized user attempts to access API without valid credentials<br>**Source:** External malicious actor<br>**Environment:** Normal operation<br>**Artifact:** API gateway<br>**Response:** System denies access and logs attempt<br>**Response Measure:** 100% of unauthorized requests are blocked, all attempts logged | UC10 | High |

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
| **QA5** | Security | API Key Authentication + Input Validation | Implement API key middleware for authentication. Validate all uploaded images (file type, size limits). Sanitize inputs to prevent injection attacks |
| **C1-C5** | Constraints | Technology Selection | Use Flask (lightweight web framework), TensorFlow 2.x (ML), OpenCV (image processing), JWT (API auth), SQLite (store API keys/history) |
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
                              |           |          |
        +---------------------+           |          +-------------------+
        |                                 |                              |
        v                                 v                              v
   [Web User]                     [Mobile User]               [Third-party Apps]
   - Upload images                - Camera access            - API integration
   - Webcam recognition           - Real-time prediction     - Batch processing
   - View history                 - Save history             - Programmatic access
   - Batch upload                                            
        |                                 |                              |
        |                                 |                              |
        v                                 v                              v
   +----------+                    +-----------+                  +------------+
   | Web      |<------------------>| REST API  |<---------------->| API        |
   | Interface|  HTTP/HTTPS        | Gateway   |  HTTP/JSON       | Clients    |
   | (Flask)  |                    | (Auth)    |                  |            |
   +----------+                    +-----------+                  +------------+
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
1. **New Actors:** Added Web User, Mobile User, and Third-party Apps
2. **API Gateway:** New component for authentication and routing
3. **Application Services Layer:** Separated business logic into modular services
4. **Database:** Added for storing user data, API keys, and prediction history
5. **Web Interface:** Flask-based UI for browser access
6. **REST API:** Enables programmatic access for integrations

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

2. **Business Logic Layer (EXPANDED):**
   - **Image Processor:** Centralized image validation and preprocessing
   - **ML Inference Engine:** Model loading with caching for performance
   - **Batch Processor:** Handles UC9 (batch processing) with queue management
   - **History Manager:** Stores and retrieves prediction history
   - **Error Handler:** Centralized exception handling and user feedback
   - **Logger:** Comprehensive logging for debugging and monitoring

3. **Data Layer (NEW):**
   - **Database (SQLite):** Persistent storage for users, API keys, and history
   - **Tables:**
     - Users: Store user accounts (if authentication is added)
     - API Keys: Store and validate API access credentials
     - History: Store prediction history for users

4. **External Dependencies:**
   - TensorFlow, OpenCV, and File System remain but are now accessed through abstraction layers

**Component Interactions:**
- Web UI and REST API both use Business Logic Layer (no direct model access)
- Authentication Middleware protects API endpoints
- All components use centralized Error Handler and Logger
- ML Inference Engine caches loaded model to improve performance (addresses QA1)
- Batch Processor uses queue for asynchronous processing (addresses QA1)

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

- **API-First Design:** Creating a REST API forced us to define clear contracts between components. This not only enabled third-party integrations (UC10) but also made our own web UI more modular since it became just another API client.

- **Error Handling Strategy:** ML models can fail in numerous ways (model file corrupted, out of memory, invalid input image). A centralized error handling component was essential to provide consistent error messages across different interfaces (CLI, Web UI, API) and log issues for debugging.

**Lesson 4: Performance Optimization Requires Architectural Trade-offs**

Real-time sign language recognition demands low latency, which influenced several architectural decisions:

- **Caching vs. Memory:** Keeping the model loaded in memory improved response time from 5 seconds to under 2 seconds, but increased baseline memory usage by ~500MB. We had to design the system to gracefully handle out-of-memory conditions and potentially unload the model during idle periods.

- **Synchronous vs. Asynchronous:** For single-image predictions (UC2, UC3), synchronous processing worked fine. But for batch processing (UC9), we needed asynchronous task queues to avoid blocking the web server. This added architectural complexity (message broker, worker processes) but was necessary for good user experience.

- **Stateless Design for Scalability:** To support multiple concurrent users (QA2), we designed the API as stateless, meaning each request is independent. This allows horizontal scaling by running multiple server instances behind a load balancer. However, it complicated features like prediction history, which required introducing a shared database.

These lessons emphasize that ML-based systems require careful architectural planning beyond traditional software concerns. The probabilistic nature of ML models, strict dependency requirements, performance constraints, and need for ongoing model improvement all shape architectural decisions in ways that developers must anticipate from the beginning of the project.

---

## Summary

This Phase 3 proposal extends the Sign Language Alphabet Recognizer with practical, implementable features:

- **New Use Cases:** Batch processing and API access enable broader usage scenarios
- **Quality Attributes:** Focus on performance, scalability, availability, modifiability, and security
- **Architectural Patterns:** Layered architecture, caching, API authentication, and error handling
- **Updated Architecture:** Clear separation of concerns with Web UI, REST API, business logic, and data layers

All proposed extensions use familiar technologies (Flask, SQLite, JWT) that can be implemented incrementally without major refactoring of existing code.
