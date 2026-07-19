const express = require("express")
const dotenv = require("dotenv")
const {ChatGoogleGenerativeAI} = require("@langchain/google-genai")
const {ChatGroq} = require("@langchain/groq")
const fs = require("fs")
const { PDFParse } = require("pdf-parse")
const {RecursiveCharacterTextSplitter} = require("@langchain/textsplitters")
const {GoogleGenerativeAIEmbeddings} = require("@langchain/google-genai")
const {TaskType} = require("@google/generative-ai")
const {QdrantVectorStore} = require("@langchain/qdrant")
const { SystemMessage , HumanMessage}=require("@langchain/core/messages")

// FROM MY RAG ASK THINKS FROM THE PDF

const port=5000
dotenv.config()
const app=express()
app.use(express.json())

const llm = new ChatGroq({
    model:"llama-3.3-70b-versatile",
    temperature:0.7,
    maxTokens:30,
    maxRetries:2
})

const embeddings = new GoogleGenerativeAIEmbeddings({
    model:"gemini-embedding-001", //768 dimensions
    taskType:TaskType.RETRIEVAL_DOCUMENT,
    title:"Document title"
})

async function getVectorStore() {
    return await QdrantVectorStore.fromExistingCollection(
        embeddings,
        {
            url: process.env.QDRANT_URL,
            collectionName: "advancebackendRAG",
        }
    );
}


const upload = async()=>{
    const pdfPath="./knowledge.pdf"
    const buffer=fs.readFileSync(pdfPath)
    const pdfResult=new PDFParse({data:buffer})
    const result=await pdfResult.getText()
    const text=result.text
    const splitter=new RecursiveCharacterTextSplitter({
        chunkSize:500,
        chunkOverlap:100
    })
    const docs=await splitter.createDocuments([text])
    const vectorStore = await getVectorStore();
    await vectorStore.addDocuments(docs)
    // console.log(docs)
    // console.log(text)

}
// upload()



app.post("/ai",async(req,res)=>{
    const {input}=req.body
    const vectorStore = await getVectorStore()
    const docs=await vectorStore.similaritySearch(input,5)
    // console.log(docs)
    const context=docs.map((d)=>d.pageContent).join("/n")
    const response = await llm.invoke(
        [
            new SystemMessage(`you are a RAGE AI assistant.
                STRIVT RULES:
                -answer only from context
                - do not use outside knowledge
                -if answer not found say:
                    "I dont know"
                    Context:${context}`),
            new HumanMessage(input)
        ]
    )
    return res.status(200).json({"ai":response.content})    
})



app.get("/",(req,res)=>{
    return res.json({message:"Hello from level 5"})
})

app.listen(port,()=>{
    console.log("connected to server on port" ,port)
})
