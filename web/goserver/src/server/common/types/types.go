package types

type BaseRequest struct {
	Request string `json:"request"`
}

type RunCommandsRequestBody struct {
	Commands []string `json:"commands"`
}

type RunCommandsRequest struct {
	BaseRequest
	Data RunCommandsRequestBody `json:"data"`
}

type ReadLogRequestBody struct {
	LogFileName string `json:"logFileName"`
}

type ReadLogRequest struct {
	BaseRequest
	Data ReadLogRequestBody `json:"data"`
}

type AMQPRequestBody struct {
	Request string      `json:"request"`
	Data    interface{} `json:"data"`
}

type FindOneRequestBody struct {
	Id string `json:"id"`
}
