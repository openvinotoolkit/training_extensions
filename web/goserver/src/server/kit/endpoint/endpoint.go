package endpoint

import (
	"context"
)

type Response struct {
	Data   interface{} `json:"data"`
	Err    interface{} `json:"err"`
	IsLast bool        `json:"isLast"`
}

type Middleware func(Endpoint) Endpoint

type Endpoint func(context.Context, interface{}) chan Response
