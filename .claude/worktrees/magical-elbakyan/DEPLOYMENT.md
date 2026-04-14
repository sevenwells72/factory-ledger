# Factory Ledger Deployment Guide

## Project Overview

Factory Ledger is a manufacturing operations system designed to replace Odoo with a conversational ChatGPT-like interface. The system maintains a complete append-only ledger of all manufacturing operations, inventory transactions, and production activities.

### Key Features
- Natural language interface (ChatGPT integration)
- - Complete lot traceability for manufactured products
  - - Inventory management with automatic updates
    - - Manufacturing order tracking and execution
      - - Append-only transaction ledger (no updates/deletes)
        - - PostgreSQL database backend (Supabase)
          - - RESTful API with authentication
           
            - ### Technology Stack
            - - **Backend**: FastAPI (Python)
              - - **Database**: PostgreSQL (Supabase)
                - - **Deployment**: Render
                  - - **Version Control**: GitHub
                    - - **Authentication**: Bearer token
                     
                      - ---

                      ## Phase 1: Pre-Deployment Setup (COMPLETED)

                      ### 1.1 Database Setup (Supabase)
                      Database created and schema initialized with 5 tables:

                      **Tables:**
                      1. **products** - Product definitions (name, unit, status)
                      2. 2. **lots** - Production lots tracking (product, quantity, dates)
                         3. 3. **transactions** - All transaction records (immutable audit log)
                            4. 4. **transaction_lines** - Individual line items (product, quantity, lot reference)
                               5. 5. **batch_formulas** - Manufacturing recipes (product composition, quantities)
                                 
                                  6. **Connection String Format:**
                                  7. ```
                                     postgresql://postgres:[PASSWORD]@[HOST].supabase.co:5432/postgres
                                     ```

                                     **Credentials:**
                                     - Host: db.vrafvwcdpcijvxdvefpr.supabase.co
                                     - - Port: 5432
                                       - - Database: postgres
                                         - - Username: postgres
                                           - - Password: [Securely stored - see user]
                                            
                                             - ### 1.2 GitHub Repository Setup (COMPLETED)
                                             - Repository: `https://github.com/sevenwells72/factory-ledger`
                                            
                                             - **Current Files:**
                                             - - `main.py` - FastAPI application (87 lines)
                                               - - `requirements.txt` - Python dependencies
                                                 - - `runtime.txt` - Python version specification (3.11.0)
                                                  
                                                   - **Key Code (main.py):**
                                                   - - 6 API endpoints
                                                     - - Environment variable configuration
                                                       - - PostgreSQL connection
                                                         - - Bearer token authentication
                                                           - - Append-only transaction logging
                                                            
                                                             - ---

                                                             ## Phase 2: Deployment to Render (IN PROGRESS)

                                                             ### 2.1 Render Account Setup

                                                             **Status:** Email verification pending

                                                             **Steps Completed:**
                                                             1. Created Render account via GitHub OAuth
                                                             2. 2. Authenticated with GitHub (sevenwells72 user)
                                                                3. 3. Email verification required at: michael@sevenwellsgranola.com
                                                                  
                                                                   4. **Next Steps:**
                                                                   5. 1. ✅ User must verify email (check inbox for verification link)
                                                                      2. 2. ✅ Grant Render permission to access GitHub repository
                                                                         3. 3. ✅ Create new Web Service in Render
                                                                            4. 4. ✅ Configure environment variables
                                                                               5. 5. ✅ Deploy application
                                                                                 
                                                                                  6. ### 2.2 Environment Variables for Render
                                                                                 
                                                                                  7. Create these environment variables in Render dashboard:
                                                                                 
                                                                                  8. | Variable | Value | Notes |
                                                                                  9. |----------|-------|-------|
                                                                                  10. | `DATABASE_URL` | `postgresql://postgres:[PASSWORD]@db.vrafvwcdpcijvxdvefpr.supabase.co:5432/postgres` | Full PostgreSQL connection string |
                                                                                  11. | `API_KEY` | `ledger-secret-2026-factory` | Bearer token for API authentication |
                                                                                  12. | `PYTHON_VERSION` | `3.11.0` | Specified in runtime.txt |
                                                                                 
                                                                                  13. ### 2.3 Render Deployment Configuration
                                                                                 
                                                                                  14. **Service Type:** Web Service (Python)
                                                                                 
                                                                                  15. **Build & Deploy Settings:**
                                                                                  16. - Start Command: `uvicorn main:app --host 0.0.0.0 --port 10000`
                                                                                      - - Python Version: 3.11.0
                                                                                        - - Build Command: `pip install -r requirements.txt`
                                                                                         
                                                                                          - **GitHub Integration:**
                                                                                          - - Repository: sevenwells72/factory-ledger
                                                                                            - - Branch: main
                                                                                              - - Auto-deploy: Enable on push to main
                                                                                               
                                                                                                - **Expected Deployment Time:** 3-5 minutes
                                                                                               
                                                                                                - **After Deployment:**
                                                                                                - - Application URL: https://[render-assigned-name].onrender.com
                                                                                                  - - All endpoints available at this URL
                                                                                                   
                                                                                                    - ---
                                                                                                    
                                                                                                    ## Phase 3: API Endpoints & Testing
                                                                                                    
                                                                                                    ### 3.1 Available Endpoints
                                                                                                    
                                                                                                    **Base URL:** `https://[render-url].onrender.com`
                                                                                                    
                                                                                                    **Headers Required:**
                                                                                                    ```
                                                                                                    Authorization: Bearer ledger-secret-2026-factory
                                                                                                    Content-Type: application/json
                                                                                                    ```
                                                                                                    
                                                                                                    ### Endpoints:
                                                                                                    
                                                                                                    #### 1. GET `/`
                                                                                                    Health check endpoint
                                                                                                    ```bash
                                                                                                    curl -H "Authorization: Bearer ledger-secret-2026-factory" \
                                                                                                      https://[render-url].onrender.com/
                                                                                                    ```
                                                                                                    
                                                                                                    #### 2. POST `/api/transaction`
                                                                                                    Record a manufacturing or inventory transaction
                                                                                                    ```bash
                                                                                                    curl -X POST \
                                                                                                      -H "Authorization: Bearer ledger-secret-2026-factory" \
                                                                                                      -H "Content-Type: application/json" \
                                                                                                      -d '{
                                                                                                        "transaction_type": "manufacture",
                                                                                                        "product_id": "classic-granola",
                                                                                                        "quantity": 300,
                                                                                                        "unit": "pounds",
                                                                                                        "notes": "Batch production - Jan 19"
                                                                                                      }' \
                                                                                                      https://[render-url].onrender.com/api/transaction
                                                                                                    ```
                                                                                                    
                                                                                                    #### 3. POST `/api/inventory-adjustment`
                                                                                                    Adjust inventory (receive, issue, count)
                                                                                                    ```bash
                                                                                                    curl -X POST \
                                                                                                      -H "Authorization: Bearer ledger-secret-2026-factory" \
                                                                                                      -H "Content-Type: application/json" \
                                                                                                      -d '{
                                                                                                        "product_id": "oats",
                                                                                                        "adjustment_type": "receive",
                                                                                                        "quantity": 500,
                                                                                                        "unit": "pounds",
                                                                                                        "reference": "PO-12345"
                                                                                                      }' \
                                                                                                      https://[render-url].onrender.com/api/inventory-adjustment
                                                                                                    ```
                                                                                                    
                                                                                                    #### 4. GET `/api/product-status/{product_id}`
                                                                                                    Get current inventory and lot information
                                                                                                    ```bash
                                                                                                    curl -H "Authorization: Bearer ledger-secret-2026-factory" \
                                                                                                      https://[render-url].onrender.com/api/product-status/classic-granola
                                                                                                    ```
                                                                                                    
                                                                                                    #### 5. POST `/api/batch-formula`
                                                                                                    Define a batch formula (recipe)
                                                                                                    ```bash
                                                                                                    curl -X POST \
                                                                                                      -H "Authorization: Bearer ledger-secret-2026-factory" \
                                                                                                      -H "Content-Type: application/json" \
                                                                                                      -d '{
                                                                                                        "product_id": "classic-granola",
                                                                                                        "ingredients": [
                                                                                                          {"product_id": "oats", "quantity": 150, "unit": "pounds"},
                                                                                                          {"product_id": "sugar", "quantity": 50, "unit": "pounds"}
                                                                                                        ]
                                                                                                      }' \
                                                                                                      https://[render-url].onrender.com/api/batch-formula
                                                                                                    ```
                                                                                                    
                                                                                                    #### 6. GET `/api/transaction-log`
                                                                                                    Retrieve complete transaction history (audit log)
                                                                                                    ```bash
                                                                                                    curl -H "Authorization: Bearer ledger-secret-2026-factory" \
                                                                                                      https://[render-url].onrender.com/api/transaction-log
                                                                                                    ```
                                                                                                    
                                                                                                    ### 3.2 Testing the Deployment
                                                                                                    
                                                                                                    After Render deployment completes:
                                                                                                    
                                                                                                    ```bash
                                                                                                    # 1. Test health check
                                                                                                    curl -H "Authorization: Bearer ledger-secret-2026-factory" \
                                                                                                      https://[render-url].onrender.com/

                                                                                                    # Expected response:
                                                                                                    # {"status": "ok", "database": "connected"}

                                                                                                    # 2. Create initial products
                                                                                                    curl -X POST \
                                                                                                      -H "Authorization: Bearer ledger-secret-2026-factory" \
                                                                                                      -H "Content-Type: application/json" \
                                                                                                      -d '{"product_id": "sugar", "name": "Sugar", "unit": "pounds"}' \
                                                                                                      https://[render-url].onrender.com/api/product

                                                                                                    # 3. Test inventory adjustment
                                                                                                    curl -X POST \
                                                                                                      -H "Authorization: Bearer ledger-secret-2026-factory" \
                                                                                                      -H "Content-Type: application/json" \
                                                                                                      -d '{"product_id": "sugar", "adjustment_type": "receive", "quantity": 1000, "unit": "pounds", "reference": "Initial stock"}' \
                                                                                                      https://[render-url].onrender.com/api/inventory-adjustment
                                                                                                    ```
                                                                                                    
                                                                                                    ---
                                                                                                    
                                                                                                    ## Phase 4: ChatGPT Custom Integration (NEXT PHASE)
                                                                                                    
                                                                                                    ### 4.1 Overview
                                                                                                    Once API is deployed, create a Custom GPT that:
                                                                                                    - Uses the Factory Ledger API
                                                                                                    - - Accepts natural language manufacturing commands
                                                                                                      - - Translates commands to API calls
                                                                                                        - - Maintains conversation context
                                                                                                          - - Requires human confirmation before database writes
                                                                                                           
                                                                                                            - ### 4.2 Implementation Steps (Future)
                                                                                                            - 1. Create Custom GPT in OpenAI
                                                                                                              2. 2. Configure API connection with authentication
                                                                                                                 3. 3. Define action schemas for manufacturing operations
                                                                                                                    4. 4. Set up confirmation workflow
                                                                                                                       5. 5. Integrate with Supabase for real-time updates
                                                                                                                         
                                                                                                                          6. ---
                                                                                                                         
                                                                                                                          7. ## Phase 5: Pilot Program
                                                                                                                         
                                                                                                                          8. ### 5.1 Pilot Team
                                                                                                                          9. - Operations manager
                                                                                                                             - - Manufacturing lead
                                                                                                                               - - Inventory manager
                                                                                                                                 - - System admin (backup)
                                                                                                                                  
                                                                                                                                   - ### 5.2 Initial Data
                                                                                                                                   - Products to seed into database:
                                                                                                                                   - - Sugar (unit: pounds)
                                                                                                                                     - - Oats (unit: pounds)
                                                                                                                                       - - Classic Granola (unit: pounds)
                                                                                                                                        
                                                                                                                                         - ### 5.3 Parallel Operations
                                                                                                                                         - - Run Factory Ledger alongside Odoo
                                                                                                                                           - - Log all operations in both systems
                                                                                                                                             - - Compare inventory accuracy daily
                                                                                                                                               - - Collect team feedback
                                                                                                                                                
                                                                                                                                                 - ### 5.4 Success Criteria
                                                                                                                                                 - - 100% transaction accuracy vs Odoo
                                                                                                                                                   - - <2 min avg response time
                                                                                                                                                     - - Team proficiency with 4 operations (receive, manufacture, adjust, query)
                                                                                                                                                       - - Zero data loss or traceability gaps
                                                                                                                                                        
                                                                                                                                                         - ---
                                                                                                                                                         
                                                                                                                                                         ## Troubleshooting
                                                                                                                                                         
                                                                                                                                                         ### Build Failures on Render
                                                                                                                                                         
                                                                                                                                                         **Problem:** Deployment fails with "Failed to build wheel for pydantic-core"
                                                                                                                                                         
                                                                                                                                                         **Solution:**
                                                                                                                                                         - Remove pydantic from requirements.txt (FastAPI includes it)
                                                                                                                                                         - - Ensure requirements.txt has no conflicting versions
                                                                                                                                                           - - Rebuild application
                                                                                                                                                            
                                                                                                                                                             - **File:** `requirements.txt`
                                                                                                                                                             - ```
                                                                                                                                                               fastapi==0.109.0
                                                                                                                                                               uvicorn[standard]==0.27.0
                                                                                                                                                               psycopg2-binary==2.9.9
                                                                                                                                                               python-dotenv==1.0.0
                                                                                                                                                               ```
                                                                                                                                                               
                                                                                                                                                               ### Database Connection Issues
                                                                                                                                                               
                                                                                                                                                               **Problem:** "Connection refused" on deployment
                                                                                                                                                               
                                                                                                                                                               **Verify:**
                                                                                                                                                               1. DATABASE_URL environment variable is set correctly
                                                                                                                                                               2. 2. Supabase IP whitelist includes Render (usually automatic)
                                                                                                                                                                  3. 3. Network connectivity: `psql -U postgres -h db.vrafvwcdpcijvxdvefpr.supabase.co`
                                                                                                                                                                    
                                                                                                                                                                     4. ### API Authentication Failures
                                                                                                                                                                    
                                                                                                                                                                     5. **Problem:** 401 Unauthorized on API calls
                                                                                                                                                                    
                                                                                                                                                                     6. **Check:**
                                                                                                                                                                     7. 1. Bearer token is exactly: `ledger-secret-2026-factory`
                                                                                                                                                                        2. 2. Header format: `Authorization: Bearer [TOKEN]`
                                                                                                                                                                           3. 3. Environment variable API_KEY matches token
                                                                                                                                                                             
                                                                                                                                                                              4. ---
                                                                                                                                                                             
                                                                                                                                                                              5. ## Important Design Decisions
                                                                                                                                                                             
                                                                                                                                                                              6. ### Append-Only Ledger
                                                                                                                                                                              7. All transactions are immutable. Corrections create NEW reversal transactions, not updates.
                                                                                                                                                                             
                                                                                                                                                                              8. **Example:**
                                                                                                                                                                              9. - Transaction 1: Received 100 lbs sugar
                                                                                                                                                                                 - - Transaction 2: Reversal - Return 50 lbs sugar (creates 50 lb negative adjustment)
                                                                                                                                                                                  
                                                                                                                                                                                   - ### Authentication
                                                                                                                                                                                   - Every API request requires Bearer token authentication. No public endpoints.
                                                                                                                                                                                  
                                                                                                                                                                                   - ### Database Schema
                                                                                                                                                                                   - - Products: Static reference data
                                                                                                                                                                                     - - Lots: Batch tracking for manufactured items
                                                                                                                                                                                       - - Transactions: Append-only audit log
                                                                                                                                                                                         - - Transaction_lines: Item-level detail
                                                                                                                                                                                           - - Batch_formulas: Recipe definitions
                                                                                                                                                                                            
                                                                                                                                                                                             - ---
                                                                                                                                                                                             
                                                                                                                                                                                             ## Monitoring & Maintenance
                                                                                                                                                                                             
                                                                                                                                                                                             ### Health Checks
                                                                                                                                                                                             Render provides:
                                                                                                                                                                                             - Automatic health checks via GET /
                                                                                                                                                                                             - - Uptime monitoring
                                                                                                                                                                                               - - Deployment logs
                                                                                                                                                                                                 - - Error alerts
                                                                                                                                                                                                  
                                                                                                                                                                                                   - ### Logs
                                                                                                                                                                                                   - Access logs in Render dashboard:
                                                                                                                                                                                                   - 1. Go to deployment
                                                                                                                                                                                                     2. 2. Click "Logs" tab
                                                                                                                                                                                                        3. 3. Filter by date/error level
                                                                                                                                                                                                          
                                                                                                                                                                                                           4. ### Database Backups
                                                                                                                                                                                                           5. Supabase provides:
                                                                                                                                                                                                           6. - Daily automated backups
                                                                                                                                                                                                              - - 7-day retention
                                                                                                                                                                                                                - - Manual backup option
                                                                                                                                                                                                                  - - Point-in-time recovery
                                                                                                                                                                                                                   
                                                                                                                                                                                                                    - ---
                                                                                                                                                                                                                    
                                                                                                                                                                                                                    ## Next Steps (Immediate)
                                                                                                                                                                                                                    
                                                                                                                                                                                                                    1. **TODAY:**
                                                                                                                                                                                                                    2.    - [ ] Verify email on Render (check michael@sevenwellsgranola.com inbox)
                                                                                                                                                                                                                          - [ ]    - [ ] Authorize Render to access GitHub repository
                                                                                                                                                                                                                          - [ ]       - [ ] Create Web Service in Render dashboard
                                                                                                                                                                                                                          - [ ]      - [ ] Configure environment variables (DATABASE_URL, API_KEY)
                                                                                                                                                                                                                          - [ ]     - [ ] Deploy application
                                                                                                                                                                                                                      
                                                                                                                                                                                                                          - [ ] 2. **TOMORROW:**
                                                                                                                                                                                                                          - [ ]    - [ ] Test all 6 API endpoints
                                                                                                                                                                                                                          - [ ]       - [ ] Seed initial product data
                                                                                                                                                                                                                          - [ ]      - [ ] Verify database connectivity
                                                                                                                                                                                                                          - [ ]     - [ ] Document API response formats
                                                                                                                                                                                                                      
                                                                                                                                                                                                                          - [ ] 3. **WEEK 1:**
                                                                                                                                                                                                                          - [ ]    - [ ] Setup Custom GPT integration
                                                                                                                                                                                                                          - [ ]       - [ ] Internal testing with operations team
                                                                                                                                                                                                                          - [ ]      - [ ] Documentation for team
                                                                                                                                                                                                                      
                                                                                                                                                                                                                          - [ ]  4. **WEEK 2+:**
                                                                                                                                                                                                                          - [ ]     - [ ] Parallel pilot with Odoo
                                                                                                                                                                                                                          - [ ]    - [ ] Team training
                                                                                                                                                                                                                          - [ ]       - [ ] Iterative improvements
                                                                                                                                                                                                                          - [ ]      - [ ] Full rollout planning
                                                                                                                                                                                                                      
                                                                                                                                                                                                                          - [ ]  ---
                                                                                                                                                                                                                      
                                                                                                                                                                                                                          - [ ]  ## Support & Questions
                                                                                                                                                                                                                      
                                                                                                                                                                                                                          - [ ]  For issues during deployment:
                                                                                                                                                                                                                          - [ ]  1. Check Render logs for error messages
                                                                                                                                                                                                                          - [ ]  2. Verify all environment variables are set
                                                                                                                                                                                                                          - [ ]  3. Confirm database connection string format
                                                                                                                                                                                                                          - [ ]  4. Test database connectivity independently
                                                                                                                                                                                                                      
                                                                                                                                                                                                                          - [ ]  For API issues:
                                                                                                                                                                                                                          - [ ]  1. Verify Bearer token is correct
                                                                                                                                                                                                                          - [ ]  2. Check request JSON format
                                                                                                                                                                                                                          - [ ]  3. Review response error messages
                                                                                                                                                                                                                          - [ ]  4. Check application logs in Render
                                                                                                                                                                                                                      
                                                                                                                                                                                                                          - [ ]  ---
                                                                                                                                                                                                                      
                                                                                                                                                                                                                          - [ ]  ## Document Version History
                                                                                                                                                                                                                      
                                                                                                                                                                                                                          - [ ]  - **v1.0** (Jan 19, 2026) - Initial comprehensive deployment guide
                                                                                                                                                                                                                          - [ ]    - Complete system overview
                                                                                                                                                                                                                          - [ ]      - Database setup documentation
                                                                                                                                                                                                                          - [ ]    - API endpoint specifications
                                                                                                                                                                                                                          - [ ]      - Deployment procedures
                                                                                                                                                                                                                          - [ ]    - Troubleshooting guide
                                                                                                                                                                                                                          - [ ]      - Next steps and timeline
