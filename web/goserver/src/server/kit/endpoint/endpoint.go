package endpoint

import (
	"context"
)

type Error struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type Response struct {
	Data   interface{} `json:"data"`
	Err    Error       `json:"err"`
	IsLast bool        `json:"isLast"`
}

type Middleware func(Endpoint) Endpoint

type Endpoint func(context.Context, interface{}) chan Response
