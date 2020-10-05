package encode_decode

type BaseAmqpRequest struct {
	Event   string `json:"event"`
	Request string `json:"request"`
}
